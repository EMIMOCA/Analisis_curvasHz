# Analisis_curvasHz
analisis de curvas hz
# app.py
from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import piexif
from scipy.optimize import curve_fit
import io
import os

app = Flask(__name__)

class CurveAnalyzer:
    def __init__(self):
        self.curves_data = []
        
    def extract_geo_data(self, image_path):
        """Extrae datos geográficos de la imagen"""
        try:
            img = Image.open(image_path)
            exif_dict = piexif.load(img.info['exif'])
            
            # Extraer coordenadas GPS si están disponibles
            if 'GPS' in exif_dict:
                gps = exif_dict['GPS']
                lat = self._convert_to_degrees(gps[2])
                lon = self._convert_to_degrees(gps[4])
                return lat, lon
        except:
            return None, None
            
    def detect_curve(self, image_path):
        """Detecta los bordes de la curva en la imagen"""
        # Leer imagen
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtros para mejorar detección
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Detectar líneas usando Transformada de Hough
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=100, maxLineGap=10)
        
        return lines, edges
        
    def fit_curve(self, points):
        """Ajusta una curva circular a los puntos detectados"""
        def circle_equation(x, h, k, r):
            return k + np.sqrt(r**2 - (x - h)**2)
            
        x = points[:, 0]
        y = points[:, 1]
        
        # Estimación inicial de parámetros
        h_init = np.mean(x)
        k_init = np.mean(y)
        r_init = np.std(x)
        
        try:
            popt, _ = curve_fit(circle_equation, x, y, 
                               p0=[h_init, k_init, r_init])
            return popt
        except:
            return None
            
    def calculate_geometry(self, curve_params, pixel_scale):
        """Calcula parámetros geométricos de la curva"""
        if curve_params is None:
            return None
            
        h, k, r = curve_params
        radius = r * pixel_scale  # Convertir píxeles a metros
        
        # Calcular grado de curvatura
        degree_curve = (18000/np.pi) / radius
        
        # Calcular longitud de curva y deflexión
        # Esto requiere detectar PC y PT
        pc_pt_distance = self._detect_pc_pt(h, k, r)
        length = pc_pt_distance * pixel_scale
        deflection = np.arctan2(pc_pt_distance, radius) * 180/np.pi
        
        return {
            'radius': radius,
            'degree_curve': degree_curve,
            'length': length,
            'deflection': deflection,
            'center': (h, k)
        }
        
    def export_to_excel(self):
        """Exporta los datos de las curvas a Excel"""
        df = pd.DataFrame(self.curves_data)
        
        # Crear archivo Excel en memoria
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Curvas', index=False)
            
            # Añadir formato
            workbook = writer.book
            worksheet = writer.sheets['Curvas']
            
            # Formato para números
            num_format = workbook.add_format({'num_format': '0.00'})
            for col in ['radius', 'degree_curve', 'length', 'deflection']:
                col_idx = df.columns.get_loc(col)
                worksheet.set_column(col_idx, col_idx, 12, num_format)
                
        output.seek(0)
        return output
        
    def _convert_to_degrees(self, gps_coords):
        """Convierte coordenadas GPS a grados decimales"""
        d = float(gps_coords[0][0]) / float(gps_coords[0][1])
        m = float(gps_coords[1][0]) / float(gps_coords[1][1])
        s = float(gps_coords[2][0]) / float(gps_coords[2][1])
        
        return d + (m / 60.0) + (s / 3600.0)
        
    def _detect_pc_pt(self, h, k, r):
        """Detecta los puntos PC y PT de la curva"""
        # Implementar detección de puntos de inicio y fin de curva
        # Esto puede requerir análisis más detallado de la imagen
        return 2 * r  # Simplificación inicial

# Rutas de la aplicación Flask
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_curve():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    image = request.files['image']
    analyzer = CurveAnalyzer()
    
    # Guardar imagen temporalmente
    temp_path = 'temp_image.jpg'
    image.save(temp_path)
    
    # Analizar curva
    lines, edges = analyzer.detect_curve(temp_path)
    if lines is None:
        return jsonify({'error': 'No curve detected'}), 400
        
    # Obtener datos geográficos
    lat, lon = analyzer.extract_geo_data(temp_path)
    
    # Ajustar curva y calcular geometría
    points = np.vstack(lines)[:, :2]  # Tomar solo puntos x,y
    curve_params = analyzer.fit_curve(points)
    
    # Usar escala aproximada basada en metadata o input del usuario
    pixel_scale = 0.1  # metros por pixel (ejemplo)
    geometry = analyzer.calculate_geometry(curve_params, pixel_scale)
    
    if geometry:
        # Agregar datos geográficos
        geometry['latitude'] = lat
        geometry['longitude'] = lon
        analyzer.curves_data.append(geometry)
        
    # Limpiar archivo temporal
    os.remove(temp_path)
    
    return jsonify(geometry)

@app.route('/export', methods=['GET'])
def export_data():
    analyzer = CurveAnalyzer()
    output = analyzer.export_to_excel()
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='curvas_geometricas.xlsx'
    )

if __name__ == '__main__':
    app.run(debug=True)
