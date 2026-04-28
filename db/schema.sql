-- Pepe Maps — MySQL schema
-- Stores Mexican states, cities (with map center coordinates) and their attractions.

SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS states (
  id    SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
  slug  VARCHAR(64)  NOT NULL,
  name  VARCHAR(128) NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY ux_states_slug (slug)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cities (
  id          INT UNSIGNED      NOT NULL AUTO_INCREMENT,
  state_id    SMALLINT UNSIGNED NOT NULL,
  slug        VARCHAR(64)       NOT NULL,
  name        VARCHAR(128)      NOT NULL,
  subtitle    VARCHAR(255)          NULL,
  icon        VARCHAR(16)           NULL,
  center_lat  DECIMAL(9,6)      NOT NULL,
  center_lng  DECIMAL(9,6)      NOT NULL,
  zoom        TINYINT UNSIGNED  NOT NULL DEFAULT 14,
  PRIMARY KEY (id),
  UNIQUE KEY ux_cities_slug (slug),
  KEY ix_cities_state (state_id),
  CONSTRAINT fk_cities_state
    FOREIGN KEY (state_id) REFERENCES states(id)
    ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS attractions (
  id           INT UNSIGNED     NOT NULL AUTO_INCREMENT,
  city_id      INT UNSIGNED     NOT NULL,
  position     TINYINT UNSIGNED NOT NULL,           -- 1..N order within the city
  name         VARCHAR(255)     NOT NULL,
  lat          DECIMAL(9,6)     NOT NULL,
  lng          DECIMAL(9,6)     NOT NULL,
  icon         VARCHAR(16)          NULL,
  category     VARCHAR(64)          NULL,
  description  TEXT                 NULL,
  history      TEXT                 NULL,
  tips         TEXT                 NULL,
  hours        VARCHAR(128)         NULL,
  rating       DECIMAL(2,1)         NULL,
  PRIMARY KEY (id),
  UNIQUE KEY ux_attractions_city_position (city_id, position),
  KEY ix_attractions_city (city_id),
  CONSTRAINT fk_attractions_city
    FOREIGN KEY (city_id) REFERENCES cities(id)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
