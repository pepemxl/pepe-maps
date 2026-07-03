<?php
// JSON API for native clients (Android app).
// GET /api/            → every city (grouped data flattened per city, attractions included)
// GET /api/?city=slug  → a single city by slug
$cities = require __DIR__ . '/../cities.php';

header('Content-Type: application/json; charset=UTF-8');
header('Access-Control-Allow-Origin: *');
header('Cache-Control: public, max-age=300');

function city_payload(string $slug, array $c): array {
    return [
        'slug'        => $slug,
        'name'        => $c['name'],
        'state'       => $c['state'] ?? 'Sin estado',
        'state_slug'  => $c['state_slug'] ?? 'sin-estado',
        'subtitle'    => $c['subtitle'] ?? '',
        'icon'        => $c['icon'] ?? '🏛️',
        'lat'         => $c['center'][0],
        'lng'         => $c['center'][1],
        'zoom'        => $c['zoom'] ?? 14,
        'attractions' => array_map(fn($a) => [
            'id'          => $a['id'],
            'name'        => $a['name'],
            'lat'         => $a['lat'],
            'lng'         => $a['lng'],
            'icon'        => $a['icon'] ?? '📍',
            'category'    => $a['category'] ?? '',
            'description' => $a['description'] ?? '',
            'history'     => $a['history'] ?? '',
            'tips'        => $a['tips'] ?? '',
            'hours'       => $a['hours'] ?? '',
            'rating'      => (float)($a['rating'] ?? 0),
        ], $c['attractions'] ?? []),
    ];
}

$slug = $_GET['city'] ?? null;

if ($slug !== null) {
    if (!isset($cities[$slug])) {
        http_response_code(404);
        echo json_encode(['error' => 'city not found'], JSON_UNESCAPED_UNICODE);
        exit;
    }
    echo json_encode(city_payload($slug, $cities[$slug]), JSON_UNESCAPED_UNICODE);
    exit;
}

$out = [];
foreach ($cities as $s => $c) {
    $out[] = city_payload($s, $c);
}
echo json_encode(['cities' => $out], JSON_UNESCAPED_UNICODE);
