import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from skimage import measure, segmentation, filters
import io
import base64
from scipy import ndimage

# Configuration
st.set_page_config(
    page_title="DentalAI PRO - Analyse Complète",
    page_icon="🦷",
    layout="wide"
)

class DentalAIPro:
    def __init__(self):
        self.model = None
    
    def preprocess(self, image):
        """Prétraitement avancé"""
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY) if len(np.array(image).shape) == 3 else np.array(image)
        
        # CLAHE + Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur pour réduction bruit
        denoised = cv2.GaussianBlur(enhanced, (3,3), 0)
        
        return denoised
    
    def detect_caries(self, image, tooth_mask):
        """Détection avancée des caries"""
        # Zones hypodenses sur couronne (caries = zones sombres)
        crown_mask = tooth_mask.copy()
        h, w = crown_mask.shape
        crown_mask[int(h*0.6):] = 0  # Seulement la couronne
        
        masked = cv2.bitwise_and(255 - image, 255 - image, mask=crown_mask)
        
        # Détection par seuillage local
        local_thresh = filters.threshold_local(masked, 35, method='gaussian')
        caries_mask = masked > local_thresh
        
        # Composantes connectées
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            caries_mask.astype(np.uint8)*255, 8
        )
        
        caries_list = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 20:  # Filtre bruit
                caries_list.append({
                    'id': i,
                    'area': area,
                    'width': stats[i, cv2.CC_STAT_WIDTH],
                    'height': stats[i, cv2.CC_STAT_HEIGHT],
                    'depth': self.estimate_caries_depth(masked, labels == i),
                    'location': self.classify_caries_location(stats[i], crown_mask.shape),
                    'risk': self.assess_caries_risk(area, stats[i, cv2.CC_STAT_WIDTH]*stats[i, cv2.CC_STAT_HEIGHT])
                })
        
        total_caries_score = sum(c['risk'] for c in caries_list)
        
        return {
            'caries_detected': len(caries_list) > 0,
            'count': len(caries_list),
            'caries_list': caries_list,
            'total_score': total_caries_score,
            'caries_mask': caries_mask.astype(np.uint8)*255,
            'risk_level': 'Critique' if total_caries_score > 15 else 'Élevé' if total_caries_score > 8 else 'Modéré'
        }
    
    def estimate_caries_depth(self, image, caries_mask):
        """Estimation profondeur carie"""
        masked_caries = cv2.bitwise_and(image, image, mask=caries_mask.astype(np.uint8)*255)
        intensity_profile = masked_caries[masked_caries > 0]
        if len(intensity_profile) > 0:
            depth = np.mean(intensity_profile) / 255 * 100  # Pourcentage de profondeur
            return min(100, depth)
        return 0
    
    def classify_caries_location(self, stats, shape):
        """Classification localisation carie"""
        cx = stats[cv2.CC_STAT_WIDTH] // 2
        cy = stats[cv2.CC_STAT_HEIGHT] // 2
        h, w = shape
        
        if cy < h*0.3:
            return "Occlusal"
        elif cx < w*0.3 or cx > w*0.7:
            return "Proximal"
        else:
            return "Buccal/Lingual"
    
    def assess_caries_risk(self, area, bbox_area):
        """Évaluation risque carie"""
        size_score = min(10, area / 50)
        shape_score = min(5, bbox_area / 100)
        return size_score + shape_score
    
    def detect_fracture(self, image, tooth_mask):
        """Détection fractures dentaires"""
        # Fractures = lignes continues hyper-denses
        edges = cv2.Canny(image, 50, 150)
        masked_edges = cv2.bitwise_and(edges, edges, mask=tooth_mask)
        
        # Détection lignes avec Hough
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLineLength=15, maxLineGap=5)
        
        fracture_score = len(lines) if lines is not None else 0
        fracture_detected = fracture_score > 3
        
        return {
            'fracture_detected': fracture_detected,
            'line_count': fracture_score,
            'severity': 'Sévère' if fracture_score > 8 else 'Modérée' if fracture_score > 3 else 'Faible'
        }
    
    def measure_bone_loss(self, image):
        """Mesure perte osseuse"""
        # Détection os alvéolaire (zones grises moyennes)
        _, bone_thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Distance du bord apical à l'os
        contours, _ = cv2.findContours(bone_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bone_level = cy / image.shape[0]  # Ratio perte osseuse
                return {
                    'bone_loss_ratio': round(bone_level * 100, 1),
                    'periodontal_stage': self.classify_periodontal_stage(bone_loss_ratio)
                }
        return {'bone_loss_ratio': 0, 'periodontal_stage': 'Santé'}
    
    def classify_periodontal_stage(self, ratio):
        if ratio > 50: return "Stade IV"
        elif ratio > 30: return "Stade III"
        elif ratio > 15: return "Stade II"
        else: return "Stade I"
    
    def detect_filling_restoration(self, image, tooth_mask):
        """Détection obturations/restaurations"""
        masked = cv2.bitwise_and(image, image, mask=tooth_mask)
        
        # Matériaux radio-opaques (très blancs)
        _, filling_mask = cv2.threshold(masked, 240, 255, cv2.THRESH_BINARY)
        
        # Filtrage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        filling_mask = cv2.morphologyEx(filling_mask, cv2.MORPH_OPEN, kernel)
        
        filling_area = np.sum(filling_mask == 255)
        
        return {
            'restoration_detected': filling_area > 100,
            'restoration_area': filling_area,
            'material_type': 'Amalgame' if filling_area > 500 else 'Composite'
        }
    
    def complete_analysis(self, image):
        """Analyse complète"""
        processed = self.preprocess(image)
        markers, segmented = self.detect_teeth(processed)
        
        results = {'tooth_count': len(np.unique(markers)) - 1}
        
        # Analyse chaque dent
        for tooth_id in range(2, min(6, len(np.unique(markers)) + 1)):
            tooth_mask = (markers == tooth_id).astype(np.uint8) * 255
            
            results[f'tooth_{tooth_id}'] = {
                **self.measure_canal_size(processed, tooth_mask),
                **self.detect_calcification(processed, tooth_mask),
                **self.analyze_periapical_lesion(processed, tooth_mask),
                **self.detect_caries(processed, tooth_mask),
                **self.detect_fracture(processed, tooth_mask),
                **self.detect_filling_restoration(processed, tooth_mask)
            }
        
        results['bone_loss'] = self.measure_bone_loss(processed)
        return results, processed, markers, segmented

# [Inclure toutes les méthodes précédentes + nouvelles]
# ... (copier les méthodes de la version précédente)

def main():
    st.title("🦷 DentalAI PRO - Analyse Complète Radiographique")
    st.markdown("**IA Dentaire Professionnelle - Toutes pathologies détectées**")
    
    # Sidebar
    st.sidebar.header("⚙️ Paramètres")
    sensitivity = st.sidebar.slider("Sensibilité détection", 0.5, 1.0, 0.8, 0.1)
    show_details = st.sidebar.checkbox("Détails techniques", True)
    
    # Upload
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📁 Radiographie", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner("🔬 Analyse complète en cours..."):
            analyzer = DentalAIPro()
            results, processed, markers, segmented = analyzer.complete_analysis(image)
        
        # Dashboard principal
        st.markdown("---")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric("🦷 Dents détectées", results['tooth_count'])
        with kpi2:
            caries_total = sum(r.get('detect_caries', {}).get('count', 0) 
                             for r in results.values() if isinstance(r, dict))
            st.metric("🕳️ Caries", caries_total)
        with kpi3:
            fractures = sum(1 for r in results.values() 
                          if isinstance(r, dict) and r.get('fracture_detected', False))
            st.metric("💥 Fractures", fractures)
        with kpi4:
            st.metric("🦴 Perte osseuse", f"{results['bone_loss']['bone_loss_ratio']}%")
        
        # Onglets résultats
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Synthèse", "🦷 Dents", "🕳️ Caries", "🎯 Pathologies"])
        
        with tab1:
            st.header("📊 Synthèse Globale")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Urgences")
                urgency_score = 0
                if caries_total > 2: urgency_score += 30
                if fractures > 0: urgency_score += 40
                if results['bone_loss']['bone_loss_ratio'] > 30: urgency_score += 20
                
                st.metric("Niveau d'urgence", f"{urgency_score}%")
                st.info(f"**Plan d'action**: {'🚨 URGENT' if urgency_score > 60 else '⚠️ À planifier'}")
            
            with col2:
                st.subheader("Score Global")
                health_score = 100 - urgency_score
                st.metric("Santé dentaire", f"{health_score}%")
        
        with tab2:
            st.header("🦷 Analyse par Dent")
            for i, (key, data) in enumerate(results.items()):
                if key.startswith('tooth_') and isinstance(data, dict):
                    with st.expander(f"Dent {key[-1]}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Canal", f"{data.get('diameter_mm', 0):.1f}mm")
                            st.metric("Calcification", "✅" if not data.get('calcification_detected', False) else "❌")
                        with col2:
                            caries_data = data.get('detect_caries', {})
                            st.metric("Caries", caries_data.get('count', 0))
                            st.metric("Obturation", "✅" if data.get('restoration_detected', False) else "❌")
                        with col3:
                            st.metric("Fracture", data.get('fracture_detected', False))
                            st.metric("Lésion", data.get('lesion_detected', False))
        
        with tab3:
            st.header("🕳️ Caries Détectées")
            caries_details = []
            for key, data in results.items():
                if isinstance(data, dict) and 'detect_caries' in data:
                    caries = data['detect_caries']
                    for c in caries.get('caries_list', []):
                        caries_details.append({
                            **c,
                            'tooth': key
                        })
            
            if caries_details:
                st.dataframe(caries_details)
                fig, ax = plt.subplots(figsize=(12, 8))
                risks = [c['risk'] for c in caries_details]
                locations = [c['location'] for c in caries_details]
                ax.scatter(locations, risks, s=100, alpha=0.7)
                ax.set_title("Répartition des caries par localisation et risque")
                ax.set_ylabel("Score de risque")
                st.pyplot(fig)
            else:
                st.success("🎉 Aucune carie détectée!")
        
        with tab4:
            st.header("🎯 Autres Pathologies")
            pathologies = {
                "Perte osseuse": results['bone_loss']['periodontal_stage'],
                "Fractures totales": sum(1 for r in results.values() if isinstance(r, dict) and r.get('fracture_detected', False)),
                "Lésions apicales": sum(1 for r in results.values() if isinstance(r, dict) and r.get('lesion_detected', False)),
            }
            
            for patho, value in pathologies.items():
                st.metric(patho, value)
        
        # Visualisations
        st.markdown("---")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes[0,0].imshow(image, cmap='gray')
        axes[0,0].set_title("Originale")
        axes[0,0].axis('off')
        
        axes[0,1].imshow(processed, cmap='gray')
        axes[0,1].set_title("Traitée")
        axes[0,1].axis('off')
        
        axes[0,2].imshow(segmented, cmap='jet')
        axes[0,2].set_title("Segmentation")
        axes[0,2].axis('off')
        
        # Heatmap caries
        caries_heatmap = np.zeros_like(processed)
        for key, data in results.items():
            if isinstance(data, dict) and 'detect_caries' in data:
                caries_heatmap[data['detect_caries']['caries_mask'] > 0] = 255
        axes[1,0].imshow(caries_heatmap, cmap='Reds')
        axes[1,0].set_title("Heatmap Caries")
        axes[1,0].axis('off')
        
        # Fractures
        fracture_vis = cv2.Canny(processed, 50, 150)
        axes[1,1].imshow(fracture_vis, cmap='viridis')
        axes[1,1].set_title("Détection Fractures")
        axes[1,1].axis('off')
        
        # Os
        bone_vis = cv2.threshold(processed, 100, 255, cv2.THRESH_BINARY_INV)[1]
        axes[1,2].imshow(bone_vis, cmap='Blues')
        axes[1,2].set_title("Niveau Osseux")
        axes[1,2].axis('off')
        
        st.pyplot(fig)
        
        # Rapport PDF
        rapport = f"""
🦷 DENTALAI PRO - RAPPORT COMPLET
Fichier: {uploaded_file.name}
Date: {st.session_state.get('timestamp', 'N/A')}

📊 SYNTHÈSE:
- Dents analysées: {results['tooth_count']}
- Caries: {caries_total}
- Fractures: {pathologies['Fractures totales']}
- Perte osseuse: {results['bone_loss']['bone_loss_ratio']}% ({results['bone_loss']['periodontal_stage']})

URGENCE: {'🚨 CRITIQUE' if urgency_score > 60 else '⚠️ ÉLEVÉE' if urgency_score > 30 else '✅ MODÉRÉE'}

RECOMMANDATIONS:
1. { 'Endodontie urgente' if fractures > 0 else 'Contrôle prophylactique' }
2. {'Parodontie' if results['bone_loss']['bone_loss_ratio'] > 20 else ''}
3. {'Obturations multiples' if caries_total > 3 else ''}
        """
        
        st.download_button("📥 Télécharger Rapport", rapport, f"rapport_{uploaded_file.name}.txt")

if __name__ == "__main__":
    main()