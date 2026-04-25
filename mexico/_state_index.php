<?php
// Per-state landing page: derives the state slug from the calling directory
// and lists every city registered under that state.

$stateSlug = basename(dirname($_SERVER['SCRIPT_FILENAME'] ?? __FILE__));
$cities = require __DIR__ . '/cities.php';

$stateName = null;
$stateCities = [];
foreach ($cities as $slug => $c) {
    if (($c['state_slug'] ?? null) === $stateSlug) {
        $stateName = $c['state'] ?? $stateSlug;
        $stateCities[$slug] = $c;
    }
}

if (!$stateCities) {
    http_response_code(404);
    header('Content-Type: text/html; charset=UTF-8');
    echo "<!doctype html><meta charset=utf-8><title>404</title>";
    echo "<p style='font-family:sans-serif;padding:40px'>Estado no encontrado: <code>"
       . htmlspecialchars($stateSlug) . "</code>. ";
    echo "<a href='/mexico/'>Ver todas las ciudades</a>.</p>";
    exit;
}

header('Content-Type: text/html; charset=UTF-8');

$totalAttractions = 0;
foreach ($stateCities as $c) { $totalAttractions += count($c['attractions'] ?? []); }
?>
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#1a0a2e">
<title>Pepe Maps · <?= htmlspecialchars($stateName) ?></title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Crimson+Pro:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    :root {
        --gold: #c9a84c;
        --gold-light: #e8c97a;
        --gold-dim: #8a6e2e;
        --deep: #0d0820;
        --midnight: #1a1035;
        --panel: rgba(13, 8, 32, 0.92);
        --surface: rgba(26, 16, 53, 0.85);
        --text: #f0e8d0;
        --text-dim: #a09078;
        --teal: #2ab5a0;
        --font-serif: 'Playfair Display', Georgia, serif;
        --font-body: 'Crimson Pro', 'Times New Roman', serif;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: var(--font-body);
        background: var(--deep);
        color: var(--text);
        min-height: 100vh;
        background-image:
            radial-gradient(circle at 10% 10%, rgba(201,168,76,0.08) 0%, transparent 50%),
            radial-gradient(circle at 90% 80%, rgba(42,181,160,0.05) 0%, transparent 50%);
    }
    header {
        text-align: center;
        padding: 60px 20px 30px;
        border-bottom: 1px solid var(--gold-dim);
    }
    header .crumb {
        color: var(--text-dim);
        font-size: 0.78rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    header .crumb a { color: var(--teal); text-decoration: none; }
    header .crumb a:hover { color: var(--gold-light); }
    header h1 {
        font-family: var(--font-serif);
        font-size: 2.4rem;
        color: var(--gold-light);
        letter-spacing: 0.05em;
        font-style: italic;
    }
    header .count {
        margin-top: 16px;
        display: inline-block;
        padding: 6px 14px;
        border: 1px solid var(--gold-dim);
        border-radius: 4px;
        color: var(--gold);
        font-size: 0.85rem;
        letter-spacing: 0.1em;
    }
    main { max-width: 1200px; margin: 0 auto; padding: 40px 20px 80px; }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 18px;
    }
    .city-card {
        background: var(--surface);
        border: 1px solid rgba(201,168,76,0.2);
        border-radius: 10px;
        padding: 22px 20px;
        text-decoration: none;
        color: inherit;
        transition: all 0.25s ease;
        display: flex;
        gap: 14px;
        align-items: flex-start;
    }
    .city-card:hover {
        border-color: var(--gold);
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(201,168,76,0.15);
    }
    .city-icon { font-size: 2rem; flex-shrink: 0; }
    .city-meta { flex: 1; }
    .city-name { font-family: var(--font-serif); font-size: 1.15rem; color: var(--gold-light); margin-bottom: 6px; }
    .city-subtitle { font-size: 0.82rem; color: var(--text-dim); margin-bottom: 10px; }
    .city-attr-count {
        display: inline-block;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--teal);
        border: 1px solid rgba(42,181,160,0.4);
        padding: 3px 8px;
        border-radius: 3px;
    }
    footer { text-align: center; padding: 30px 20px 60px; color: var(--text-dim); font-size: 0.78rem; letter-spacing: 0.1em; }
    @media (max-width: 480px) { header h1 { font-size: 1.8rem; } main { padding: 24px 14px 60px; } }
</style>
</head>
<body>
    <header>
        <div class="crumb"><a href="../">← México</a></div>
        <h1><?= htmlspecialchars($stateName) ?></h1>
        <span class="count"><?= count($stateCities) ?> ciudades · <?= $totalAttractions ?> atracciones</span>
    </header>
    <main>
        <div class="grid">
            <?php foreach ($stateCities as $slug => $c):
                $name = htmlspecialchars($c['name']);
                $sub  = htmlspecialchars($c['subtitle'] ?? '');
                $icon = $c['icon'] ?? '🏛️';
                $attrCount = count($c['attractions'] ?? []);
            ?>
            <a class="city-card" href="<?= rawurlencode($slug) ?>/">
                <div class="city-icon"><?= $icon ?></div>
                <div class="city-meta">
                    <div class="city-name"><?= $name ?></div>
                    <?php if ($sub): ?><div class="city-subtitle"><?= $sub ?></div><?php endif; ?>
                    <span class="city-attr-count"><?= $attrCount ?> atracciones</span>
                </div>
            </a>
            <?php endforeach; ?>
        </div>
    </main>
    <footer>Pepe Maps · Open data · OpenStreetMap · OSRM</footer>
</body>
</html>
