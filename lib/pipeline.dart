import 'dart:convert';
import 'dart:io';

/// Runs the Python/MLX pipeline steps as subprocesses, streaming their output to
/// [onLog]. Flutter is only the control plane — MLX does the work.
class Pipeline {
  final String studioRoot;
  final String python; // interpreter (use the .venv one)
  final String baseModel;
  final String llamaCppDir;
  final String quant;
  final void Function(String line) onLog;

  Process? _current;
  bool get running => _current != null;

  Pipeline({
    required this.studioRoot,
    required this.python,
    required this.baseModel,
    required this.llamaCppDir,
    required this.quant,
    required this.onLog,
  });

  Future<int> _run(String exe, List<String> args, String label) async {
    if (running) {
      onLog('⚠ a step is already running — wait for it to finish.');
      return -1;
    }
    onLog('\n▶ $label\n  \$ $exe ${args.join(' ')}');
    try {
      final p = await Process.start(exe, args,
          workingDirectory: studioRoot, runInShell: false);
      _current = p;
      p.stdout.transform(utf8.decoder).transform(const LineSplitter()).listen(
          onLog);
      p.stderr.transform(utf8.decoder).transform(const LineSplitter()).listen(
          onLog);
      final code = await p.exitCode;
      _current = null;
      onLog(code == 0 ? '✓ $label done' : '✗ $label exited with $code');
      return code;
    } catch (e) {
      _current = null;
      onLog('✗ $label failed to start: $e');
      return -1;
    }
  }

  void cancel() {
    _current?.kill(ProcessSignal.sigterm);
    _current = null;
    onLog('■ cancelled');
  }

  Future<int> checkSupport() =>
      _run(python, ['py/inspect_model.py', baseModel], 'Check support (inspect model)');

  Future<int> prepareData() =>
      _run(python, ['py/prepare_data.py'], 'Prepare data');

  Future<int> train() =>
      _run('bash', ['py/train.sh', baseModel, 'py/lora_config.yaml'], 'Train (LoRA)');

  Future<int> evaluate() =>
      _run(python, ['py/eval.py', '--model', 'workspace/fused'], 'Eval (fused)');

  Future<int> exportGguf() => _run(
      'bash', ['py/export_gguf.sh', baseModel, llamaCppDir, quant], 'Export GGUF');
}
