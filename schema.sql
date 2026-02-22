-- SmartX Vision Platform v3 — Database Schema
-- All tables enforce company_id isolation

CREATE TABLE IF NOT EXISTS vision_ppe_config (
    id INT AUTO_INCREMENT PRIMARY KEY,
    company_id INT NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    display_name VARCHAR(100),
    body_region VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_company_class (company_id, class_name),
    INDEX idx_company (company_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS vision_detection_events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    company_id INT NOT NULL,
    compliant BOOLEAN DEFAULT FALSE,
    missing_items JSON,
    detections JSON,
    faces JSON,
    snapshot_path VARCHAR(500),
    person_code VARCHAR(50),
    person_name VARCHAR(255),
    camera_id INT,
    zone_id INT,
    edge_device_id VARCHAR(100),
    model_name VARCHAR(100),
    confidence_threshold FLOAT,
    processing_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_company_time (company_id, created_at DESC),
    INDEX idx_compliant (company_id, compliant),
    INDEX idx_person (company_id, person_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS vision_alerts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    company_id INT NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity ENUM('low','medium','high','critical') DEFAULT 'medium',
    details JSON,
    person_code VARCHAR(50),
    camera_id INT,
    zone_id INT,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP NULL,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_company_time (company_id, created_at DESC),
    INDEX idx_severity (company_id, severity, resolved)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS vision_people (
    id INT AUTO_INCREMENT PRIMARY KEY,
    company_id INT NOT NULL,
    person_code VARCHAR(50) NOT NULL,
    person_name VARCHAR(255) NOT NULL,
    badge_id VARCHAR(100),
    department VARCHAR(100),
    face_photos_count INT DEFAULT 0,
    face_embedding_path VARCHAR(500),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_company_person (company_id, person_code),
    INDEX idx_company (company_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS vision_edge_devices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    company_id INT NOT NULL,
    device_id VARCHAR(100) NOT NULL,
    device_type VARCHAR(50),
    location VARCHAR(255),
    ip_address VARCHAR(45),
    gpu_model VARCHAR(100),
    status ENUM('online','offline','maintenance') DEFAULT 'offline',
    last_heartbeat TIMESTAMP NULL,
    config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_device (company_id, device_id),
    INDEX idx_company_status (company_id, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS vision_training_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    company_id INT NOT NULL,
    base_model VARCHAR(50),
    epochs INT,
    batch_size INT,
    img_size INT,
    classes JSON,
    status ENUM('pending','training','complete','error') DEFAULT 'pending',
    best_map50 FLOAT,
    best_map50_95 FLOAT,
    model_path VARCHAR(500),
    error_message TEXT,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_company (company_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS vision_video_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    company_id INT NOT NULL,
    source_type VARCHAR(20),
    source_url VARCHAR(1000),
    frames_total INT DEFAULT 0,
    frames_compliant INT DEFAULT 0,
    compliance_rate FLOAT,
    detect_faces BOOLEAN DEFAULT FALSE,
    result JSON,
    status ENUM('pending','processing','complete','error') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_company (company_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
