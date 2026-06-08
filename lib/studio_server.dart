import 'dart:convert';
import 'dart:io';

import 'model_scanner.dart';
import 'pipeline.dart';

/// Embedded HTTPS (self-signed) API. Lets the Nexus agents push training data
/// AND lets a remote driver run the whole pipeline (prepare/train/eval/export,
/// HF download/upload) over HTTP while the desktop UI streams it live. Falls back
/// to plain HTTP if openssl/cert isn't available.
class StudioServer {
  final String studioRoot;
  final int port;
  final ModelScanner scanner;
  final Pipeline pipeline;
  final List<String> Function() logsTail;
  final void Function(String stage) onStage;
  final void Function(String line) onLog;

  HttpServer? _server;
  bool secure = false;

  StudioServer({
    required this.studioRoot,
    required this.port,
    required this.scanner,
    required this.pipeline,
    required this.logsTail,
    required this.onStage,
    required this.onLog,
  });

  String get rawPath => '$studioRoot/workspace/data/raw.jsonl';
  String get conversationsDir => '$studioRoot/workspace/data/conversations';
  String get baseUrl => '${secure ? 'https' : 'http'}://localhost:$port';

  bool _writeConversation(String id, List<dynamic> messages) {
    final dir = Directory(conversationsDir)..createSync(recursive: true);
    final safe = id.replaceAll(RegExp(r'[^A-Za-z0-9_.-]'), '_');
    final f = File('${dir.path}/$safe.json');
    if (f.existsSync()) {
      try {
        final prev = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
        final prevLen = (prev['messages'] as List?)?.length ?? 0;
        if (messages.length <= prevLen) return false;
      } catch (_) {}
    }
    f.writeAsStringSync(jsonEncode({'messages': messages}));
    return true;
  }

  Future<void> start() async {
    Directory('$studioRoot/workspace/data').createSync(recursive: true);
    final ctx = await _buildTls('$studioRoot/workspace/certs');
    try {
      _server = ctx != null
          ? await HttpServer.bindSecure(InternetAddress.loopbackIPv4, port, ctx)
          : await HttpServer.bind(InternetAddress.loopbackIPv4, port);
      secure = ctx != null;
    } catch (e) {
      onLog('Server failed to bind on $port: $e');
      return;
    }
    onLog('API listening at $baseUrl  (${secure ? "TLS" : "plain HTTP"})');
    _server!.listen(_handle);
  }

  Future<void> stop() async => _server?.close(force: true);

  Future<SecurityContext?> _buildTls(String certDir) async {
    final cert = '$certDir/cert.pem';
    final key = '$certDir/key.pem';
    Directory(certDir).createSync(recursive: true);
    if (!File(cert).existsSync() || !File(key).existsSync()) {
      try {
        final r = await Process.run('openssl', [
          'req', '-x509', '-newkey', 'rsa:2048',
          '-keyout', key, '-out', cert,
          '-days', '3650', '-nodes', '-subj', '/CN=localhost',
        ]);
        if (r.exitCode != 0) {
          onLog('openssl cert gen failed; using plain HTTP.');
          return null;
        }
      } catch (e) {
        onLog('openssl not available ($e); using plain HTTP.');
        return null;
      }
    }
    try {
      return SecurityContext()
        ..useCertificateChain(cert)
        ..usePrivateKey(key);
    } catch (e) {
      onLog('TLS context failed ($e); using plain HTTP.');
      return null;
    }
  }

  Future<void> _handle(HttpRequest req) async {
    try {
      final path = req.uri.path;
      final method = req.method;

      if (method == 'GET' && path == '/health') return _json(req, {'ok': true});
      if (method == 'GET' && path == '/status') {
        return _json(req, {
          'running': pipeline.running,
          'env_ready': pipeline.envReady,
          'examples': _rawCount(),
        });
      }
      if (method == 'GET' && path == '/logs') {
        final tail = int.tryParse(req.uri.queryParameters['tail'] ?? '') ?? 100;
        final all = logsTail();
        final slice = all.length > tail ? all.sublist(all.length - tail) : all;
        return _json(req, {'lines': slice, 'total': all.length});
      }
      if (method == 'GET' && path == '/models') {
        final models = await scanner.scan();
        return _json(req, {'models': [for (final m in models) m.toJson()]});
      }
      if (method == 'GET' && path == '/repos') {
        return _json(req, await pipeline.listRepos());
      }
      if (method == 'GET' && path == '/search') {
        final q = req.uri.queryParameters['q'] ?? '';
        return _json(req, {'results': await pipeline.searchModels(q)});
      }
      if (method == 'GET' && path == '/search-datasets') {
        final q = req.uri.queryParameters['q'] ?? '';
        return _json(req, {'results': await pipeline.searchDatasets(q)});
      }

      // ── Seed data (editable JSON) ────────────────────────────────────────
      if (method == 'GET' && path == '/seeds') {
        return _json(req, {'names': pipeline.seedNames()});
      }
      if (method == 'GET' && path.startsWith('/seeds/')) {
        final name = path.substring('/seeds/'.length);
        return _json(req, {'name': name, 'content': pipeline.readSeed(name)});
      }
      if (method == 'POST' && path.startsWith('/seeds/')) {
        final name = path.substring('/seeds/'.length);
        final obj = await _body(req);
        final err = pipeline.writeSeed(name, '${obj['content'] ?? ''}');
        if (err != null) return _bad(req, err);
        return _json(req, {'saved': true, 'name': name});
      }

      // ── Dataset table CRUD ───────────────────────────────────────────────
      if (method == 'GET' && path == '/data') {
        final limit = int.tryParse(req.uri.queryParameters['limit'] ?? '') ?? 500;
        final offset = int.tryParse(req.uri.queryParameters['offset'] ?? '') ?? 0;
        return _json(req, {
          'rows': pipeline.datasetSummary(limit: limit, offset: offset),
          'total': pipeline.datasetCount(),
        });
      }
      if (method == 'POST' && path == '/data/add') {
        final obj = await _body(req);
        final msgs = obj['messages'];
        if (msgs is! List) return _bad(req, 'messages[] required');
        final ok = pipeline.addConversation(msgs,
            source: (obj['source'] ?? 'manual').toString(),
            tools: obj['tools'] is List ? obj['tools'] as List : null);
        onLog(ok ? '+ dataset row added' : '· dataset row rejected (dup/invalid)');
        return _json(req, {'added': ok, 'total': pipeline.datasetRows().length});
      }
      if (method == 'POST' && path == '/data/delete') {
        final obj = await _body(req);
        final id = (obj['id'] ?? '').toString();
        if (id.isEmpty) return _bad(req, 'id required');
        final ok = pipeline.deleteConversation(id);
        onLog(ok ? '- dataset row deleted' : '· id not found');
        return _json(req, {'deleted': ok, 'total': pipeline.datasetRows().length});
      }

      if (method == 'POST' && path == '/training-data') {
        final obj = await _body(req);
        final msgs = obj['messages'];
        if (msgs is! List) return _bad(req, 'messages[] required');
        final cid = obj['conversation_id'];
        if (cid is String && cid.isNotEmpty) {
          final wrote = _writeConversation(cid, msgs);
          onLog(wrote
              ? '+ conversation "$cid" (${msgs.length} msgs)'
              : '· conversation "$cid" unchanged');
          return _json(req, {'conversation': cid, 'messages': msgs.length});
        }
        _appendItems([{'messages': msgs}]);
        onLog('+ training trace (total ${_rawCount()})');
        return _json(req, {'added': 1, 'total': _rawCount()});
      }

      // ── Remote pipeline control ──────────────────────────────────────────
      if (method == 'POST' && path == '/run/cancel') {
        pipeline.cancel();
        onStage('idle');
        return _json(req, {'cancelled': true});
      }
      if (method == 'POST' && path.startsWith('/run/')) {
        if (pipeline.running) return _busy(req);
        final step = path.substring('/run/'.length);
        final obj = await _body(req);
        Future<int>? fut;
        switch (step) {
          case 'setup-env':
            fut = pipeline.setupEnv();
            break;
          case 'check-support':
            fut = pipeline.checkSupport();
            break;
          case 'prepare':
            fut = pipeline.prepareData();
            break;
          case 'generate':
            fut = pipeline.generateData(
                kinds: (obj['kinds'] ?? 'setup,discovery,tasks').toString());
            break;
          case 'import-excel':
            final p = (obj['path'] ?? '').toString();
            if (p.isEmpty) return _bad(req, 'path required');
            fut = pipeline.importExcel(p);
            break;
          case 'template':
            fut = pipeline.exportTemplate(
                (obj['path'] ?? '$studioRoot/workspace/data/template.xlsx')
                    .toString());
            break;
          case 'import-hf':
            final ds = (obj['dataset'] ?? '').toString();
            if (ds.isEmpty) return _bad(req, 'dataset required');
            fut = pipeline.importHf(ds,
                split: (obj['split'] ?? 'train').toString(),
                config: obj['config']?.toString(),
                limit: int.tryParse('${obj['limit'] ?? ''}'));
            break;
          case 'quantize-base':
            fut = pipeline.quantizeBase();
            break;
          case 'eval':
            fut = pipeline.evaluate();
            break;
          case 'test':
            fut = pipeline.runTests(model: obj['model'] as String?);
            break;
          case 'eval-tools':
            fut = pipeline.evalToolCalls(
                model: obj['model'] as String?,
                limit: int.tryParse('${obj['limit'] ?? 100}') ?? 100);
            break;
          case 'export':
            fut = pipeline.exportGguf();
            break;
          case 'train':
            fut = pipeline.train(
              model: obj['model'] as String?,
              iters: int.tryParse('${obj['iters'] ?? ''}'),
              defaultKeys: obj['default_keys'] == true,
            );
            break;
          case 'download':
            final repo = (obj['repo'] ?? '').toString();
            if (repo.isEmpty) return _bad(req, 'repo required');
            fut = pipeline.downloadModel(repo);
            break;
          case 'upload':
            fut = pipeline.uploadModel((obj['src'] ?? '').toString(),
                (obj['dest'] ?? '').toString(), obj['private'] == true);
            break;
          default:
            return _bad(req, 'unknown step "$step"');
        }
        onStage(step);
        fut.whenComplete(() => onStage('idle'));
        return _json(req, {'started': step});
      }

      req.response.statusCode = HttpStatus.notFound;
      await req.response.close();
    } catch (e) {
      _bad(req, '$e');
    }
  }

  Future<Map<String, dynamic>> _body(HttpRequest req) async {
    final s = await utf8.decoder.bind(req).join();
    if (s.trim().isEmpty) return {};
    final d = jsonDecode(s);
    return d is Map<String, dynamic> ? d : {};
  }

  int _appendItems(List<dynamic> items) {
    final f = File(rawPath);
    final buf = StringBuffer();
    var added = 0;
    for (final it in items) {
      if (it is Map && it['messages'] is List) {
        buf.writeln(jsonEncode({'messages': it['messages']}));
        added++;
      }
    }
    if (added > 0) f.writeAsStringSync(buf.toString(), mode: FileMode.append);
    return added;
  }

  int _rawCount() {
    final f = File(rawPath);
    if (!f.existsSync()) return 0;
    return f.readAsLinesSync().where((l) => l.trim().isNotEmpty).length;
  }

  void _json(HttpRequest req, Object data) {
    req.response.headers.contentType = ContentType.json;
    req.response.write(jsonEncode(data));
    req.response.close();
  }

  void _bad(HttpRequest req, String msg) {
    req.response.statusCode = HttpStatus.badRequest;
    req.response.write(jsonEncode({'error': msg}));
    req.response.close();
  }

  void _busy(HttpRequest req) {
    req.response.statusCode = HttpStatus.conflict;
    req.response.write(jsonEncode({'error': 'a step is already running'}));
    req.response.close();
  }
}
