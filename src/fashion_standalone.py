"""Compiled presentation/retry controller; CPU tested, CUDA requires Aqua smoke test.

Only 784 input rates and N counters cross the GPU boundary per attempt.
Weights remain on the device, except bounded inference snapshots. Standalone
snapshots are NOT exact resume checkpoints; use the runtime backend for resume.
"""
import contextlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np


@contextlib.contextmanager
def _cuda_repeated_run_patches(device):
    """Make Brian2CUDA's generated code correct for our compiled image loop.

    Brian2CUDA assumes every Python-level ``run`` call executes ONCE. Here the
    two runs (present = run 0, rest = run 1) are repeated inside a C++ loop for
    every image, which breaks two things in the generated sources:

    1. rand.cu: ``RandomNumberBuffer`` selects its buffers by ``run_counter``
       (0, 1, ...). From the third run on (``run_counter >= 2``) no buffer is
       refilled or advanced, so the Poisson input reuses the SAME 784 uniform
       numbers at every time step of every later image: each pixel then fires
       either at every step or never. Fix: after both runs have initialised
       their buffers once, cycle ``run_counter`` 0,1,0,1,... and force a refill
       from the (continuing) cuRAND stream at the start of every run. No new
       device memory is allocated, so nothing leaks.
    2. before_run_*_push_spikes.cu (run before EVERY run) append to a global
       memory report and print all of it plus INFO lines every run, so the
       log grows with the square of the number of runs. Reset the report each
       run and print only on the first run.
    """
    import brian2cuda.device as cuda_device

    writer_class = cuda_device.CUDAWriter
    original_write = writer_class.write
    patched = {'rand': 0, 'before_run': 0}

    def rng_patch(source):
        pattern = re.compile(r'void RandomNumberBuffer::run_finished\(\)\s*\{\s*'
                             r'needs_init = true;\s*run_counter \+= 1;\s*\}')
        if len(pattern.findall(source)) != 1:
            raise RuntimeError('Unexpected Brian2CUDA rand.cu layout; refusing to run '
                               'the compiled loop with frozen random numbers')
        per_run = device.codeobjects_with_rng['host_api']['per_run']
        cases = []
        for run_index, by_type in enumerate(per_run):
            lines = [f'idx_{kind}_{co.name} = {kind}_interval_{co.name};'
                     for kind, objects in sorted(by_type.items()) for co in objects]
            cases.append(f'        case {run_index}: {" ".join(lines)} break;')
        body = '\n'.join(cases)
        replacement = f'''void RandomNumberBuffer::run_finished()
{{
    // CS6886 patch (see src/fashion_standalone.py): the generated runs are
    // repeated in a compiled loop, so cycle through them and refill buffers.
    static bool fashion_all_runs_initialised = false;
    run_counter += 1;
    if (run_counter >= {len(per_run)})
        fashion_all_runs_initialised = true;
    if (!fashion_all_runs_initialised)
    {{
        needs_init = true;
        return;
    }}
    run_counter %= {len(per_run)};
    needs_init = false;
    switch (run_counter)
    {{
{body}
        default: break;
    }}
}}'''
        patched['rand'] += 1
        return pattern.sub(lambda _: replacement, source)

    def write(self, filename, contents):
        if isinstance(contents, str):
            if filename == 'rand.cu':
                contents = rng_patch(contents)
            elif re.fullmatch(r'code_objects/before_run_\w+_push_spikes\.cu', filename):
                anchor = 'static bool first_run = true;'
                if contents.count(anchor) != 1 or 'std::vector<tuple_t> memory_recorder;' not in contents:
                    raise RuntimeError(f'Unexpected Brian2CUDA layout in {filename}; refusing to build')
                # memory_recorder is a file-level vector that grows every run and
                # is printed in full every run: output grows with the square of
                # the number of runs (the 2nd GPU pilot did <100 images in 2 h).
                # Start each run with an empty recorder and silence std::cout
                # (only used for these reports here) after the first run.
                contents = contents.replace(anchor, anchor + '''
    memory_recorder.clear();
    struct FashionQuietCout {
        bool quiet;
        explicit FashionQuietCout(bool q) : quiet(q) { if (quiet) std::cout.setstate(std::ios::failbit); }
        ~FashionQuietCout() { if (quiet) std::cout.clear(); }
    } fashion_quiet_cout(!first_run);''')
                contents = contents.replace('printf("INFO ', 'if (first_run) printf("INFO ')
                patched['before_run'] += 1
        return original_write(self, filename, contents)

    writer_class.write = write
    try:
        yield
    finally:
        writer_class.write = original_write
    if patched['rand'] != 1 or patched['before_run'] == 0:
        raise RuntimeError(f'Brian2CUDA sources not patched as expected ({patched}); '
                           'refusing to trust the CUDA run')


def run_standalone(a, arrays, manifest, schedule, root, signature):
    from brian2 import defaultclock, get_device, prefs, set_device
    from .fashion_data import save_json
    from .fashion_runner import summarize
    from .network import build_network, ETA
    from .encoding import PRESENT_TIME, REST_TIME
    from .utils import save_weights

    cuda = a.backend == 'cuda'
    if cuda:
        # Require a bounded pilot until generated CUDA control has run on Aqua.
        pilot_sized = a.train_limit and a.train_limit <= 1000 and a.epochs == 1
        if not pilot_sized and not getattr(a, 'allow_long_cuda', False):
            raise ValueError('CUDA is experimental: first use --train-limit <=1000 --epochs 1; '
                             'after that pilot succeeds on the GPU, pass --allow-long-cuda')
        import brian2cuda  # noqa: F401 registers the device
        prefs.devices.cuda_standalone.cuda_backend.gpu_id = a.gpu_id
        # Driver 470 cannot answer newer nvidia-smi compute-capability queries.
        prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
        prefs.devices.cuda_standalone.cuda_backend.compute_capability = 7.0
        set_device('cuda_standalone', build_on_run=False)
    else:
        set_device('cpp_standalone', build_on_run=False)
        prefs.devices.cpp_standalone.openmp_threads = a.threads if a.threads > 1 else 0
    prefs.devices.cpp_standalone.extra_make_args_unix = [f'-j{a.threads}']
    prefs.codegen.cpp.headers += ['<fstream>', '<vector>', '<chrono>', '<stdexcept>', '<algorithm>', '<cmath>']
    components = build_network(a.n_excitatory, a.seed)
    device = get_device()

    def queue_run(duration):
        # Brian normally inserts Python-time clock constants before every run.
        # Inside this compiled loop those constants would rewind the first tick
        # of every image, corrupting event-driven pretraces. Replace ONLY these
        # clock initializations with the current compiled network time.
        offset = len(device.main_queue)
        components['net'].run(duration)
        names = {device.get_array_name(defaultclock.variables['t']): 'fashion_network.t',
                 device.get_array_name(defaultclock.variables['timestep']):
                 'static_cast<int64_t>(std::llround(fashion_network.t / '
                 + device.get_array_name(defaultclock.variables['dt']) + '[0]))'}
        changed = set()
        for index in range(offset, len(device.main_queue)):
            kind, values = device.main_queue[index]
            if kind == 'set_by_single_value' and values[0] in names:
                device.main_queue[index] = (kind, (values[0], values[1], names[values[0]]))
                changed.add(values[0])
        if changed != set(names):
            raise RuntimeError('Unsupported Brian clock queue layout; refusing unsafe compiled loop')
    # Key the build folder by dataset folder AND run folder: two workers in one
    # job (e.g. MNIST and Fashion-MNIST with identical settings) share
    # FASHION_BUILD_ROOT and have identical run-folder names.
    work_root = (Path(os.environ['FASHION_BUILD_ROOT']) / root.parent.name / root.name
                 if 'FASHION_BUILD_ROOT' in os.environ else root)
    work_root.mkdir(parents=True, exist_ok=True)
    binary = work_root / 'inputs'
    binary.mkdir()
    for name, values in (('train', arrays[0]), ('test', arrays[2]),
                         ('ids', schedule['ids'].astype('<i8')), ('stage', schedule['stage'])):
        values.tofile(binary / f'{name}.bin')

    def host(obj, key):
        var = obj.variables[key]
        if var.dynamic:
            return device.get_array_name(var, access_data=False) + '.data()'
        return device.get_array_name(var)

    def sync(obj, key, size, to_device):
        if not cuda:
            return ''
        var = obj.variables[key]
        if var.dynamic:
            # Synaptic variables are growable: Brian2CUDA stores their device copy
            # in a thrust::device_vector named dev_dynamic_array_..., not in a raw
            # dev_array_... pointer (that name does not exist and fails to compile).
            vector = 'dev' + device.get_array_name(var, access_data=False)
            pointer = f'thrust::raw_pointer_cast({vector}.data())'
        else:
            pointer = device.get_array_name(var, prefix='dev')
        hp = host(obj, key)
        dst, src = (pointer, hp) if to_device else (hp, pointer)
        direction = 'cudaMemcpyHostToDevice' if to_device else 'cudaMemcpyDeviceToHost'
        return (f'if(cudaMemcpy({dst}, {src}, {size} * sizeof({np.dtype(var.dtype).name.replace("float64", "double").replace("int32", "int")}), '
                f'{direction}) != cudaSuccess) throw std::runtime_error("CUDA copy failed: {key}");')

    rates = host(components['input'], 'rates')
    mon = host(components['exc_monitor'], 'count')
    eta = host(components['syn_input_exc'], 'eta')
    plasticity = host(components['exc'], 'plasticity_enabled')
    weights = host(components['syn_input_exc'], 'w')
    theta = host(components['exc'], 'theta')
    ne = a.n_excitatory
    ns = 784 * ne
    count_sync = sync(components['exc_monitor'], 'count', ne, False)
    rate_sync = sync(components['input'], 'rates', 784, True)
    phase_sync = (sync(components['syn_input_exc'], 'eta', ns, True) +
                  sync(components['exc'], 'plasticity_enabled', 1, True))
    snapshot_sync = (sync(components['syn_input_exc'], 'w', ns, False) +
                     sync(components['exc'], 'theta', ne, False))
    snapshot_dir = root / 'inference_snapshots'
    snapshot_dir.mkdir()
    # C++ string literals are quoted with JSON, never interpolated into a shell.
    quoted = lambda path: json.dumps(str(path))
    ntrain = int(np.count_nonzero(schedule['stage'] == 0))
    nrecords = len(schedule['ids'])
    device.insert_code('main', f'''
    auto load_bytes = [](const char* path, size_t count) {{
        std::vector<unsigned char> values(count);
        std::ifstream file(path, std::ios::binary);
        if(!file.read(reinterpret_cast<char*>(values.data()), count))
            throw std::runtime_error("Cannot read input bytes");
        return values;
    }};
    auto fashion_train = load_bytes({quoted(binary / 'train.bin')}, {arrays[0].size});
    auto fashion_test = load_bytes({quoted(binary / 'test.bin')}, {arrays[2].size});
    auto fashion_stage = load_bytes({quoted(binary / 'stage.bin')}, {nrecords});
    std::vector<int64_t> fashion_ids({nrecords});
    std::ifstream fashion_id_file({quoted(binary / 'ids.bin')}, std::ios::binary);
    if(!fashion_id_file.read(reinterpret_cast<char*>(fashion_ids.data()), {nrecords} * sizeof(int64_t)))
        throw std::runtime_error("Cannot read schedule");
    std::ofstream fashion_responses({quoted(root / 'counts.bin')}, std::ios::binary);
    if(!fashion_responses) throw std::runtime_error("Cannot write response file");
    double fashion_seconds[3] = {{0, 0, 0}};
    long long fashion_attempts[6] = {{0, 0, 0, 0, 0, 0}};
    long long fashion_exhausted = 0;
    int fashion_previous_stage = -1;
    std::vector<int> fashion_before({ne}), fashion_counts({ne});
    auto fashion_snapshot = [&](size_t position) {{
        {snapshot_sync}
        for(size_t k=0; k < {ns}; ++k)
            if(!std::isfinite({weights}[k])) throw std::runtime_error("Nonfinite trained weights");
        std::string prefix = std::string({quoted(snapshot_dir)}) + "/position_" + std::to_string(position);
        std::ofstream wfile(prefix + "_w.bin", std::ios::binary);
        std::ofstream tfile(prefix + "_theta.bin", std::ios::binary);
        wfile.write(reinterpret_cast<const char*>({weights}), {ns} * sizeof(double));
        tfile.write(reinterpret_cast<const char*>({theta}), {ne} * sizeof(double));
        if(!wfile || !tfile) throw std::runtime_error("Cannot write inference snapshot");
    }};
    for(size_t fashion_position = 0; fashion_position < {nrecords}; ++fashion_position) {{
        int fashion_phase = fashion_stage[fashion_position];
        auto fashion_start = std::chrono::steady_clock::now();
        if(fashion_phase != fashion_previous_stage) {{
            std::fill({eta}, {eta} + {ns}, fashion_phase == 0 ? {ETA} : 0.0);
            {plasticity}[0] = fashion_phase == 0 ? 1.0 : 0.0;
            {phase_sync}
            fashion_previous_stage = fashion_phase;
        }}
        const unsigned char* fashion_pixels =
            (fashion_phase == 2 && {int(not a.validation_size)}) ? fashion_test.data() : fashion_train.data();
        int fashion_total = 0, fashion_attempt = 0;
        for(; fashion_attempt < 6; ++fashion_attempt) {{
            for(int k = 0; k < 784; ++k)
                {rates}[k] = fashion_pixels[fashion_ids[fashion_position] * 784 + k] / 4.0
                              * (1.0 + fashion_attempt * 32.0 / 63.75);
            {rate_sync}
            {count_sync}
            std::copy({mon}, {mon} + {ne}, fashion_before.begin());
    ''')
    queue_run(PRESENT_TIME)
    device.insert_code('main', f'''
            {count_sync}
            fashion_total = 0;
            for(int k = 0; k < {ne}; ++k) {{
                fashion_counts[k] = {mon}[k] - fashion_before[k];
                fashion_total += fashion_counts[k];
            }}
            std::fill({rates}, {rates} + 784, 0.0);
            {rate_sync}
    ''')
    queue_run(REST_TIME)
    device.insert_code('main', f'''
            if(fashion_total >= 5) break;
        }}
        ++fashion_attempts[std::min(fashion_attempt, 5)];
        if(fashion_total < 5) ++fashion_exhausted;
        fashion_seconds[fashion_phase] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - fashion_start).count();
        if(fashion_phase != 0) {{
            fashion_responses.write(reinterpret_cast<const char*>(fashion_counts.data()), {ne} * sizeof(int));
            if(!fashion_responses) throw std::runtime_error("Response write failed");
        }}
        if((fashion_position + 1) % 100 == 0)
            std::cout << "progress=" << fashion_position + 1 << "/{nrecords}" << std::endl;
        if(fashion_phase == 0 && ((fashion_position + 1) % {len(manifest['train'])} == 0
               || (fashion_position + 1) % 5000 == 0))
            fashion_snapshot(fashion_position + 1);
    }}
    fashion_responses.close();
    std::ofstream fashion_metrics({quoted(root / 'standalone_timing.json')});
    fashion_metrics << "{{\\\"phase_seconds\\\":[" << fashion_seconds[0] << "," << fashion_seconds[1]
        << "," << fashion_seconds[2] << "],\\\"attempts\\\":[";
    for(int k = 0; k < 6; ++k) fashion_metrics << (k ? "," : "") << fashion_attempts[k];
    fashion_metrics << "],\\\"exhausted\\\":" << fashion_exhausted << "}}";
    if(!fashion_metrics) throw std::runtime_error("Cannot write timing file");
    ''')
    build_start = time.monotonic()
    save_json(root / 'status.json', dict(status='building', backend=a.backend,
              standalone_resume='unsupported', cuda_validation='experimental' if cuda else 'not applicable'))
    if cuda:
        with _cuda_repeated_run_patches(device):
            device.build(directory=str(work_root / 'build'), compile=True, run=not a.compile_only)
    else:
        device.build(directory=str(work_root / 'build'), compile=True, run=not a.compile_only)
    build_and_run_seconds = time.monotonic() - build_start
    if a.compile_only:
        save_json(root / 'status.json', dict(status='compiled', run=False))
        return 0
    counts = np.fromfile(root / 'counts.bin', dtype=np.int32).reshape(nrecords - ntrain, ne)
    progress = json.loads((root / 'standalone_timing.json').read_text())
    progress.update(position=nrecords, signature=signature,
                    build_and_run_seconds=build_and_run_seconds, standalone_resume='unsupported')
    save_weights(components, root / 'baseline_weights.npz')
    trained_w = np.fromfile(snapshot_dir / f'position_{ntrain}_w.bin')
    trained_theta = np.fromfile(snapshot_dir / f'position_{ntrain}_theta.bin')
    with np.load(root / 'baseline_weights.npz') as frozen:
        if (not np.array_equal(trained_w, frozen['w']) or
                not np.array_equal(trained_theta, frozen['theta_volts'])):
            raise RuntimeError('Frozen standalone observation changed weights/thresholds')
    summarize(a, arrays, manifest, counts, root, progress)
    return 0
