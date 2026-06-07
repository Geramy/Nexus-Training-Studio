import 'dart:convert';
import 'dart:io';

import 'model_scanner.dart';

/// Embedded HTTPS (self-signed) API the Nexus agents call to push training data.
/// Falls back to plain HTTP if openssl/cert isn't available.
class StudioServer {
  final String studioRoot;
  final int port;
  final ModelScanner scanner;
  final String Function() statusGetter;
  final void Function(String line) onLog;

  HttpServer? _server;
  bool secure = false;

  StudioServer({
    required this.studioRoot,
    required this.port,
    required this.scanner,
    required this.statusGetter,
    required this.onLog,
  });

  String get rawPath => '$studioRoot/workspace/data/raw.jsonl';
  String get conversationsDir => '$studioRoot/workspace/data/conversations';
  String get baseUrl => '${secure ? 'https' : 'http'}://localhost:$port';

  /// Write one conversation as its own file, keyed by id, keeping the LONGEST
  /// version (a re-post with fewer messages is ignored). Returns true if written.
  bool _writeConversation(String id, List<dynamic> messages) {
    final dir = Directory(conversationsDir)..createSync(recursive: true);
    final safe = id.replaceAll(RegExp(r'[^A-Za-z0-9_.-]'), '_');
    final f = File('${dir.path}/$safe.json');
    if (f.existsSync()) {
      try {
        final prev = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
        final prevLen = (prev['messages'] as List?)?.length ?? 0;
        if (messages.length <= prevLen) return false; // keep the longer one
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
          ? await HttpServer.bindSecure(
              InternetAddress.loopbackIPv4, port, ctx)
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
          onLog('openssl cert gen failed; using plain HTTP. ${r.stderr}');
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
      if (req.method == 'GET' && path == '/health') {
        return _json(req, {'ok': true});
      }
      if (req.method == 'GET' && path == '/status') {
        return _json(req, {'status': statusGetter(), 'examples': _rawCount()});
      }
      if (req.method == 'GET' && path == '/models') {
        final models = await scanner.scan();
        return _json(req, {'models': [for (final m in models) m.toJson()]});
      }
      if (req.method == 'POST' && path == '/training-data') {
        final body = await utf8.decoder.bind(req).join();
        final obj = jsonDecode(body) as Map<String, dynamic>;
        final msgs = obj['messages'];
        if (msgs is! List) {
          req.response.statusCode = HttpStatus.badRequest;
          req.response.write(jsonEncode({'error': 'messages[] required'}));
          return req.response.close();
        }
        final cid = obj['conversation_id'];
        if (cid is String && cid.isNotEmpty) {
          // Keep the LONGEST trace per conversation — a growing interview
          // re-posts each turn; the final, complete version wins.
          final wrote = _writeConversation(cid, msgs);
          onLog(wrote
              ? '+ conversation "$cid" (${msgs.length} msgs)'
              : '· conversation "$cid" unchanged (shorter re-post ignored)');
          return _json(req, {'conversation': cid, 'messages': msgs.length});
        }
        final n = _appendItems([{'messages': msgs}]);
        onLog('+ received 1 training trace (total ${_rawCount()})');
        return _json(req, {'added': n, 'total': _rawCount()});
      }
      if (req.method == 'POST' && path == '/training-data/batch') {
        final body = await utf8.decoder.bind(req).join();
        final obj = jsonDecode(body) as Map<String, dynamic>;
        final items = (obj['items'] as List? ?? const []);
        final n = _appendItems(items);
        onLog('+ received $n training traces (total ${_rawCount()})');
        return _json(req, {'added': n, 'total': _rawCount()});
      }
      req.response.statusCode = HttpStatus.notFound;
      await req.response.close();
    } catch (e) {
      req.response.statusCode = HttpStatus.badRequest;
      req.response.write(jsonEncode({'error': '$e'}));
      await req.response.close();
    }
  }

  /// Each item must be `{"messages":[...]}`. Returns how many were accepted.
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
    return f
        .readAsLinesSync()
        .where((l) => l.trim().isNotEmpty)
        .length;
  }

  void _json(HttpRequest req, Object data) {
    req.response.headers.contentType = ContentType.json;
    req.response.write(jsonEncode(data));
    req.response.close();
  }
}
