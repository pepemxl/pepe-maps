# pepe-maps

Interactive PWA tourist maps for the **50 most important cities in Mexico**, built with PHP, Leaflet and OSRM walking routes.

## Quick start

```bash
php -S 127.0.0.1:8000 -t mexico
# then open http://127.0.0.1:8000/
```

The landing page lists every city; click one to open its map.

## Layout

```
mexico/
├── index.php                       # Landing page (cities grouped by state)
├── template.php                    # Shared HTML/CSS/JS renderer
├── cities.php                      # Registry: 50 cities × 5 attractions
├── _render.php                     # City dispatcher (state + city slugs)
├── _state_index.php                # State landing page renderer
└── <state-slug>/
    ├── index.php                   # One-liner → _state_index.php
    └── <city-slug>/index.php       # One-liner → _render.php
                                     # e.g. mexico/jalisco/guadalajara/index.php
```

URLs follow `/<state-slug>/<city-slug>/` (e.g. `/yucatan/merida/`).
Requesting `/<state>/` shows every city in that state.

To **add a new city**:

1. Append an entry to `mexico/cities.php` including `'state'` and `'state_slug'`.
2. Create `mexico/<state-slug>/<city-slug>/index.php` with:

   ```php
   <?php require __DIR__ . '/../../_render.php';
   ```

3. If it's the first city for that state, also create
   `mexico/<state-slug>/index.php` with:

   ```php
   <?php require __DIR__ . '/../_state_index.php';
   ```

## What you get per city

- Dark colonial-themed Leaflet map (CartoDB Dark tiles)
- Custom emoji markers for every attraction
- Live geolocation (red pulse) with walking-route ETA via OSRM
- Rich popups with description, history, traveller tips, hours, rating
- Bottom carousel of attraction cards
- PWA: per-city manifest, service worker, installable
