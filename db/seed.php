<?php
/**
 * Pepe Maps — DB seeder.
 *
 * Reads mexico/cities.php and upserts states, cities and attractions into MySQL.
 * Idempotent: safe to run repeatedly.
 *
 * Connection is read from env vars (with docker-compose-friendly defaults):
 *   DB_HOST  (default: 127.0.0.1)
 *   DB_PORT  (default: 3306)
 *   DB_NAME  (default: pepe_maps)
 *   DB_USER  (default: pepe)
 *   DB_PASS  (default: pepe)
 *
 * Usage:
 *   php db/seed.php
 */

declare(strict_types=1);

$root = dirname(__DIR__);
$registry = require $root . '/mexico/cities.php';

$host = getenv('DB_HOST') ?: '127.0.0.1';
$port = (int)(getenv('DB_PORT') ?: 3306);
$name = getenv('DB_NAME') ?: 'pepe_maps';
$user = getenv('DB_USER') ?: 'pepe';
$pass = getenv('DB_PASS') ?: 'pepe';

$dsn = "mysql:host={$host};port={$port};dbname={$name};charset=utf8mb4";

try {
    $pdo = new PDO($dsn, $user, $pass, [
        PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_EMULATE_PREPARES   => false,
    ]);
} catch (PDOException $e) {
    fwrite(STDERR, "Cannot connect to MySQL at {$host}:{$port}/{$name}: {$e->getMessage()}\n");
    exit(1);
}

$pdo->beginTransaction();

$upsertState = $pdo->prepare(
    'INSERT INTO states (slug, name) VALUES (:slug, :name)
     ON DUPLICATE KEY UPDATE name = VALUES(name), id = LAST_INSERT_ID(id)'
);

$upsertCity = $pdo->prepare(
    'INSERT INTO cities (state_id, slug, name, subtitle, icon, center_lat, center_lng, zoom)
     VALUES (:state_id, :slug, :name, :subtitle, :icon, :lat, :lng, :zoom)
     ON DUPLICATE KEY UPDATE
        state_id   = VALUES(state_id),
        name       = VALUES(name),
        subtitle   = VALUES(subtitle),
        icon       = VALUES(icon),
        center_lat = VALUES(center_lat),
        center_lng = VALUES(center_lng),
        zoom       = VALUES(zoom),
        id         = LAST_INSERT_ID(id)'
);

$upsertAttraction = $pdo->prepare(
    'INSERT INTO attractions
        (city_id, position, name, lat, lng, icon, category, description, history, tips, hours, rating)
     VALUES
        (:city_id, :position, :name, :lat, :lng, :icon, :category, :description, :history, :tips, :hours, :rating)
     ON DUPLICATE KEY UPDATE
        name        = VALUES(name),
        lat         = VALUES(lat),
        lng         = VALUES(lng),
        icon        = VALUES(icon),
        category    = VALUES(category),
        description = VALUES(description),
        history     = VALUES(history),
        tips        = VALUES(tips),
        hours       = VALUES(hours),
        rating      = VALUES(rating)'
);

$stateCount = 0;
$cityCount  = 0;
$attrCount  = 0;

foreach ($registry as $citySlug => $city) {
    $upsertState->execute([
        ':slug' => $city['state_slug'],
        ':name' => $city['state'],
    ]);
    $stateId = (int)$pdo->lastInsertId();
    $stateCount++;

    [$lat, $lng] = $city['center'];
    $upsertCity->execute([
        ':state_id' => $stateId,
        ':slug'     => $citySlug,
        ':name'     => $city['name'],
        ':subtitle' => $city['subtitle'] ?? null,
        ':icon'     => $city['icon'] ?? null,
        ':lat'      => $lat,
        ':lng'      => $lng,
        ':zoom'     => $city['zoom'] ?? 14,
    ]);
    $cityId = (int)$pdo->lastInsertId();
    $cityCount++;

    foreach ($city['attractions'] ?? [] as $a) {
        $upsertAttraction->execute([
            ':city_id'     => $cityId,
            ':position'    => $a['id'],
            ':name'        => $a['name'],
            ':lat'         => $a['lat'],
            ':lng'         => $a['lng'],
            ':icon'        => $a['icon']     ?? null,
            ':category'    => $a['category'] ?? null,
            ':description' => $a['description'] ?? null,
            ':history'     => $a['history']  ?? null,
            ':tips'        => $a['tips']     ?? null,
            ':hours'       => $a['hours']    ?? null,
            ':rating'      => $a['rating']   ?? null,
        ]);
        $attrCount++;
    }
}

$pdo->commit();

$states = (int)$pdo->query('SELECT COUNT(*) FROM states')->fetchColumn();
$cities = (int)$pdo->query('SELECT COUNT(*) FROM cities')->fetchColumn();
$attrs  = (int)$pdo->query('SELECT COUNT(*) FROM attractions')->fetchColumn();

printf("Seeded: processed %d state rows, %d cities, %d attractions.\n",
    $stateCount, $cityCount, $attrCount);
printf("Totals in DB: %d states, %d cities, %d attractions.\n",
    $states, $cities, $attrs);
