<?php
// Auto-dispatcher: derives the city slug from the calling directory's name
// and the state slug from its parent directory, looks the city up in
// cities.php, validates state ↔ city pairing, and renders the shared template.

$scriptDir   = dirname($_SERVER['SCRIPT_FILENAME'] ?? __FILE__);
$citySlug    = basename($scriptDir);
$stateSlug   = basename(dirname($scriptDir));

$cities = require __DIR__ . '/cities.php';

if (!isset($cities[$citySlug])) {
    http_response_code(404);
    header('Content-Type: text/html; charset=UTF-8');
    echo "<!doctype html><meta charset=utf-8><title>404</title>";
    echo "<p style='font-family:sans-serif;padding:40px'>Ciudad no encontrada: <code>"
       . htmlspecialchars($citySlug) . "</code>. ";
    echo "<a href='/mexico/'>Ver todas las ciudades</a>.</p>";
    exit;
}

$expectedState = $cities[$citySlug]['state_slug'] ?? null;
if ($expectedState !== null && $stateSlug !== $expectedState) {
    http_response_code(301);
    header('Location: /mexico/' . rawurlencode($expectedState) . '/' . rawurlencode($citySlug) . '/');
    exit;
}

$city = $cities[$citySlug] + ['slug' => $citySlug];
require __DIR__ . '/template.php';
