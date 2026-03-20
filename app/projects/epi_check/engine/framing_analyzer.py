"""
SmartX Vision - Framing Analyzer
Análise automática de enquadramento de câmera para detecção de EPIs.

Avalia:
- Distância adequada (pessoas visíveis e identificáveis)
- Ângulo da câmera (frontal, lateral, superior)
- Cobertura da área (zona útil para detecção)
- Qualidade da detecção (taxa de sucesso)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class FramingScore:
    """Resultado da análise de enquadramento"""
    overall_score: float  # 0-100
    distance_score: float  # 0-100
    angle_score: float  # 0-100
    coverage_score: float  # 0-100
    detection_score: float  # 0-100
    
    recommendations: List[str]
    issues: List[str]
    is_acceptable: bool  # True se score >= 70
    
    # Métricas detalhadas
    avg_person_height: float  # Altura média das pessoas detectadas (pixels)
    person_count: int
    head_visibility: float  # % de cabeças visíveis
    angle_estimate: str  # "frontal", "lateral", "superior", "inferior"


class FramingAnalyzer:
    """Analisador de enquadramento de câmera para EPIs"""
    
    # Thresholds ideais
    IDEAL_PERSON_HEIGHT_MIN = 150  # pixels (muito longe)
    IDEAL_PERSON_HEIGHT_MAX = 600  # pixels (muito perto)
    IDEAL_PERSON_HEIGHT_OPTIMAL = 300  # pixels (ideal)
    
    MIN_ACCEPTABLE_SCORE = 70.0
    
    def __init__(self):
        self.history = []
        self.max_history = 30  # Últimos 30 frames
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        frame_width: int,
        frame_height: int
    ) -> FramingScore:
        """
        Analisa o enquadramento baseado em um frame e suas detecções.
        
        Args:
            frame: Frame da câmera (BGR)
            detections: Lista de detecções do YOLO [{bbox, class, conf}, ...]
            frame_width: Largura do frame
            frame_height: Altura do frame
            
        Returns:
            FramingScore com análise completa
        """
        
        # Separar detecções de pessoas e EPIs
        people = [d for d in detections if d.get('class') in ['person', 'Person']]
        epis = [d for d in detections if d.get('class') not in ['person', 'Person']]
        
        # 1. Análise de distância (baseada no tamanho das pessoas)
        distance_score, avg_height = self._analyze_distance(people, frame_height)
        
        # 2. Análise de ângulo (baseada na posição das cabeças)
        angle_score, angle_type = self._analyze_angle(people, frame_height)
        
        # 3. Análise de cobertura (área útil do frame)
        coverage_score = self._analyze_coverage(people, frame_width, frame_height)
        
        # 4. Análise de qualidade de detecção
        detection_score, head_vis = self._analyze_detection_quality(people, epis)
        
        # Score geral (média ponderada)
        overall_score = (
            distance_score * 0.3 +
            angle_score * 0.25 +
            coverage_score * 0.20 +
            detection_score * 0.25
        )
        
        # Gerar recomendações
        recommendations, issues = self._generate_recommendations(
            distance_score, angle_score, coverage_score, detection_score,
            avg_height, angle_type, len(people)
        )
        
        result = FramingScore(
            overall_score=round(overall_score, 1),
            distance_score=round(distance_score, 1),
            angle_score=round(angle_score, 1),
            coverage_score=round(coverage_score, 1),
            detection_score=round(detection_score, 1),
            recommendations=recommendations,
            issues=issues,
            is_acceptable=overall_score >= self.MIN_ACCEPTABLE_SCORE,
            avg_person_height=round(avg_height, 1),
            person_count=len(people),
            head_visibility=round(head_vis, 1),
            angle_estimate=angle_type
        )
        
        # Armazenar no histórico
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return result
    
    def _analyze_distance(
        self,
        people: List[Dict],
        frame_height: int
    ) -> Tuple[float, float]:
        """
        Analisa se as pessoas estão a uma distância adequada.
        Retorna (score, altura_média)
        """
        if not people:
            return 0.0, 0.0
        
        heights = []
        for person in people:
            bbox = person.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                height = y2 - y1
                heights.append(height)
        
        if not heights:
            return 0.0, 0.0
        
        avg_height = np.mean(heights)
        
        # Score baseado na proximidade do ideal
        if avg_height < self.IDEAL_PERSON_HEIGHT_MIN:
            # Muito longe
            score = (avg_height / self.IDEAL_PERSON_HEIGHT_MIN) * 60
        elif avg_height > self.IDEAL_PERSON_HEIGHT_MAX:
            # Muito perto
            excess = avg_height - self.IDEAL_PERSON_HEIGHT_MAX
            score = max(0, 60 - (excess / 10))
        else:
            # Na faixa aceitável
            diff = abs(avg_height - self.IDEAL_PERSON_HEIGHT_OPTIMAL)
            max_diff = self.IDEAL_PERSON_HEIGHT_MAX - self.IDEAL_PERSON_HEIGHT_OPTIMAL
            score = 100 - (diff / max_diff) * 40
        
        return min(100, max(0, score)), avg_height
    
    def _analyze_angle(
        self,
        people: List[Dict],
        frame_height: int
    ) -> Tuple[float, str]:
        """
        Analisa o ângulo da câmera baseado na posição das cabeças.
        Retorna (score, tipo_ângulo)
        """
        if not people:
            return 0.0, "unknown"
        
        # Calcular posição vertical média das pessoas
        vertical_positions = []
        for person in people:
            bbox = person.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # Posição da cabeça (topo da bbox)
                head_pos = y1 / frame_height
                vertical_positions.append(head_pos)
        
        if not vertical_positions:
            return 0.0, "unknown"
        
        avg_head_pos = np.mean(vertical_positions)
        
        # Classificar ângulo
        if avg_head_pos < 0.2:
            angle_type = "superior"  # Câmera muito alta
            score = 50
        elif avg_head_pos > 0.7:
            angle_type = "inferior"  # Câmera muito baixa
            score = 60
        elif 0.3 <= avg_head_pos <= 0.5:
            angle_type = "frontal"  # Ideal
            score = 100
        else:
            angle_type = "levemente_inclinado"
            score = 80
        
        return score, angle_type
    
    def _analyze_coverage(
        self,
        people: List[Dict],
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Analisa se a área de interesse está bem coberta.
        """
        if not people:
            return 0.0
        
        # Calcular área ocupada pelas pessoas
        total_area = frame_width * frame_height
        occupied_area = 0
        
        for person in people:
            bbox = person.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                occupied_area += w * h
        
        coverage_ratio = occupied_area / total_area
        
        # Ideal: 10-40% do frame ocupado por pessoas
        if 0.10 <= coverage_ratio <= 0.40:
            score = 100
        elif coverage_ratio < 0.10:
            # Muito vazio (câmera longe ou poucas pessoas)
            score = (coverage_ratio / 0.10) * 70
        else:
            # Muito cheio (câmera perto demais)
            score = max(0, 70 - (coverage_ratio - 0.40) * 100)
        
        return min(100, max(0, score))
    
    def _analyze_detection_quality(
        self,
        people: List[Dict],
        epis: List[Dict]
    ) -> Tuple[float, float]:
        """
        Analisa a qualidade das detecções.
        Retorna (score, head_visibility)
        """
        if not people:
            return 0.0, 0.0
        
        # 1. Verificar confiança das detecções
        confidences = [p.get('confidence', 0) for p in people]
        avg_conf = np.mean(confidences) if confidences else 0
        
        # 2. Taxa de EPIs por pessoa (indicador de qualidade)
        epi_ratio = len(epis) / len(people) if people else 0
        
        # 3. Estimar visibilidade de cabeças (baseado em helmets detectados)
        helmets = [e for e in epis if 'helmet' in e.get('class', '').lower()]
        head_visibility = (len(helmets) / len(people) * 100) if people else 0
        
        # Score combinado
        conf_score = avg_conf * 100
        epi_score = min(100, epi_ratio * 50)  # Ideal ~2 EPIs por pessoa
        
        score = (conf_score * 0.6 + epi_score * 0.4)
        
        return score, head_visibility
    
    def _generate_recommendations(
        self,
        distance_score: float,
        angle_score: float,
        coverage_score: float,
        detection_score: float,
        avg_height: float,
        angle_type: str,
        person_count: int
    ) -> Tuple[List[str], List[str]]:
        """Gera recomendações e problemas identificados"""
        
        recommendations = []
        issues = []
        
        # Análise de distância
        if distance_score < 70:
            if avg_height < self.IDEAL_PERSON_HEIGHT_MIN:
                issues.append("Câmera muito distante - pessoas pequenas demais")
                recommendations.append("🔍 Aproxime a câmera ou use lente com zoom")
                recommendations.append("📏 Ideal: pessoas com ~300px de altura no frame")
            elif avg_height > self.IDEAL_PERSON_HEIGHT_MAX:
                issues.append("Câmera muito próxima - pessoas grandes demais")
                recommendations.append("🔙 Afaste a câmera para enquadrar melhor")
                recommendations.append("📐 Objetivo: capturar corpo inteiro das pessoas")
        
        # Análise de ângulo
        if angle_score < 70:
            if angle_type == "superior":
                issues.append("Ângulo muito superior - câmera muito alta")
                recommendations.append("⬇️ Abaixe a câmera para ângulo mais frontal")
                recommendations.append("🎯 Ideal: capturar rostos e EPIs frontalmente")
            elif angle_type == "inferior":
                issues.append("Ângulo muito inferior - câmera muito baixa")
                recommendations.append("⬆️ Levante a câmera para melhor visualização")
        
        # Análise de cobertura
        if coverage_score < 70:
            issues.append("Área de cobertura inadequada")
            recommendations.append("📊 Ajuste posição para 10-40% do frame ocupado")
        
        # Análise de detecção
        if detection_score < 70:
            issues.append("Baixa qualidade de detecção")
            recommendations.append("💡 Melhore iluminação da área")
            recommendations.append("🎥 Verifique foco e limpeza da lente")
        
        # Sem pessoas
        if person_count == 0:
            issues.append("Nenhuma pessoa detectada")
            recommendations.append("👥 Verifique se há pessoas na área")
            recommendations.append("📹 Confirme se a câmera está apontada corretamente")
        
        # Tudo OK
        if not issues:
            recommendations.append("✅ Enquadramento adequado para detecção de EPIs")
        
        return recommendations, issues
    
    def get_average_score(self, last_n: int = 10) -> Optional[FramingScore]:
        """Retorna score médio dos últimos N frames"""
        if not self.history:
            return None
        
        recent = self.history[-last_n:]
        
        avg_result = FramingScore(
            overall_score=np.mean([s.overall_score for s in recent]),
            distance_score=np.mean([s.distance_score for s in recent]),
            angle_score=np.mean([s.angle_score for s in recent]),
            coverage_score=np.mean([s.coverage_score for s in recent]),
            detection_score=np.mean([s.detection_score for s in recent]),
            recommendations=[],
            issues=[],
            is_acceptable=np.mean([s.overall_score for s in recent]) >= self.MIN_ACCEPTABLE_SCORE,
            avg_person_height=np.mean([s.avg_person_height for s in recent]),
            person_count=int(np.mean([s.person_count for s in recent])),
            head_visibility=np.mean([s.head_visibility for s in recent]),
            angle_estimate=recent[-1].angle_estimate if recent else "unknown"
        )
        
        return avg_result


# Instância global
framing_analyzer = FramingAnalyzer()
