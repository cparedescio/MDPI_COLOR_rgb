import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QGridLayout, QHBoxLayout,
    QFileDialog, QSpinBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento de Imágenes")
        self.img_original = None
        self.img_procesada = None

        # Labels
        self.label_original = QLabel("Imagen Original")
        self.label_procesada = QLabel("Imagen Procesada")
        self.label_original.setFixedSize(640, 480)
        self.label_procesada.setFixedSize(640, 480)
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_procesada.setAlignment(Qt.AlignCenter)
        self.btn_guardar = QPushButton("Guardar imagen")
        # Botones
        btn_cargar = QPushButton("Cargar Imagen")
        btn_original = QPushButton("Copiar Original")
        btn_bitmixing = QPushButton("Bitmixing")
        btn_erosion = QPushButton("Erosión")
        btn_dilatacion = QPushButton("Dilatación")
        btn_apertura = QPushButton("Apertura")
        btn_cerradura = QPushButton("Cerradura")
        btn_apertura_rec = QPushButton("Apertura por Reconstrucción")
        btn_cerradura_rec = QPushButton("Cerradura por Reconstrucción")
        btn_zonas_planas = QPushButton("Zonas planas")
  #      btn_zonas_planas2 = QPushButton("Zonas planas 2")
        btn_segmentacion = QPushButton("Segmentación por Color")
        btn_gradiente_interno = QPushButton("Gradiente Interno")
        btn_gradiente_externo = QPushButton("Gradiente Externo")
        btn_gradiente_total = QPushButton("Gradiente Completo")

        # SpinBox para tamaño del kernel
        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(1, 50)
        self.kernel_size.setValue(5)
        self.btn_guardar.clicked.connect(self.guardar_imagen)

        # Eventos
        btn_cargar.clicked.connect(self.cargar_imagen)
        btn_original.clicked.connect(self.original)
        btn_bitmixing.clicked.connect(self.aplicar_bitmixing)
        btn_erosion.clicked.connect(lambda: self.aplicar_morfologia("erosion"))
        btn_dilatacion.clicked.connect(lambda: self.aplicar_morfologia("dilatacion"))
        btn_apertura.clicked.connect(lambda: self.aplicar_morfologia("apertura"))
        btn_cerradura.clicked.connect(lambda: self.aplicar_morfologia("cerradura"))
        btn_zonas_planas.clicked.connect(self.detectar_zonas_planas)
 #       btn_zonas_planas2.clicked.connect(self.detectar_zonas_planas2)
        btn_apertura_rec.clicked.connect(lambda: self.aplicar_reconstruccion("apertura"))
        btn_cerradura_rec.clicked.connect(lambda: self.aplicar_reconstruccion("cerradura"))
        btn_segmentacion.clicked.connect(self.segmentacion_color)
        btn_gradiente_interno.clicked.connect(self.aplicar_gradiente_interno)
        btn_gradiente_externo.clicked.connect(self.aplicar_gradiente_externo)
        btn_gradiente_total.clicked.connect(self.aplicar_gradiente_total)
        
        # Layout
        layout_main = QVBoxLayout()
        layout_imgs = QHBoxLayout()
        layout_imgs.addWidget(self.label_original)
        layout_imgs.addWidget(self.label_procesada)

        layout_botones = QGridLayout()     
        layout_botones.addWidget(btn_cargar, 0, 0)
        layout_botones.addWidget(btn_original, 0, 1)
        layout_botones.addWidget(btn_bitmixing, 0, 2)
        layout_botones.addWidget(btn_erosion, 1, 0)
        layout_botones.addWidget(btn_dilatacion, 1, 1)
        layout_botones.addWidget(btn_apertura, 1, 2)
        layout_botones.addWidget(btn_cerradura, 1, 3)
        layout_botones.addWidget(btn_apertura_rec, 2, 0)
        layout_botones.addWidget(btn_cerradura_rec, 2, 1)
        layout_botones.addWidget(btn_zonas_planas, 2, 2)
#        layout_botones.addWidget(btn_zonas_planas2, 2, 3)
        layout_botones.addWidget(btn_segmentacion, 3, 0)
        layout_botones.addWidget(btn_gradiente_interno, 3, 1)
        layout_botones.addWidget(btn_gradiente_externo, 3, 2)
        layout_botones.addWidget(btn_gradiente_total, 3, 3)
        layout_botones.addWidget(self.kernel_size, 4, 0)
        layout_botones.addWidget(self.btn_guardar, 4, 1)
        
        layout_main.addLayout(layout_imgs)
        layout_main.addLayout(layout_botones)
        self.setLayout(layout_main)

    def cargar_imagen(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            self.img_original = cv2.imread(file_name)
            self.img_procesada = self.img_original.copy()
            self.mostrar_imagen(self.label_original, self.img_original)

    def original(self):
        b, g, r = cv2.split(self.img_original)
        self.img_procesada = cv2.merge([b, g, r])
        self.mostrar_imagen(self.label_procesada, self.img_procesada)
    
    def mostrar_imagen(self, label, img):
        if img is None:
            return
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio))

    def aplicar_bitmixing(self):
        if self.img_original is None:
            return
        b, g, r = cv2.split(self.img_original)
        # Orden verde, rojo, azul (G,R,B)
        self.img_procesada = cv2.merge([g, r, b])
        self.mostrar_imagen(self.label_procesada, self.img_procesada)
    
    def guardar_imagen(self):
        if self.img_procesada is None:
            print("No hay imagen procesada para guardar.")
            return
    # Abrir diálogo para elegir ubicación y nombre
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar imagen",
            "",
            "Imagen PNG (*.png);;Imagen JPG (*.jpg);;Todos los archivos (*)",
            options=options
            )    
        if file_path:
            cv2.imwrite(file_path, self.img_procesada)
            print(f"Imagen guardada en: {file_path}")

    def aplicar_morfologia(self, operacion):
        if self.img_procesada is None:
            return
        k = self.kernel_size.value()
        kernel = np.ones((k, k), np.uint8)
        if operacion == "erosion":
            proc = cv2.erode(self.img_procesada, kernel, iterations=1)
        elif operacion == "dilatacion":
            proc = cv2.dilate(self.img_procesada, kernel, iterations=1)
        elif operacion == "apertura":
            proc = cv2.morphologyEx(self.img_procesada, cv2.MORPH_OPEN, kernel)
        elif operacion == "cerradura":
            proc = cv2.morphologyEx(self.img_procesada, cv2.MORPH_CLOSE, kernel)
        self.img_procesada = proc
        self.mostrar_imagen(self.label_procesada, self.img_procesada)

    def aplicar_reconstruccion(self, tipo):
        if self.img_procesada is None:
            return
        k = self.kernel_size.value()
        kernel = np.ones((k, k), np.uint8)
        if tipo == "apertura":
            marcador = cv2.erode(self.img_procesada, kernel)
            self.img_procesada = cv2.dilate(marcador, kernel)
        elif tipo == "cerradura":
            marcador = cv2.dilate(self.img_procesada, kernel)
            self.img_procesada = cv2.erode(marcador, kernel)
        self.mostrar_imagen(self.label_procesada, self.img_procesada)

    def segmentacion_color(self):
        if self.img_procesada is None:
            return
        hsv = cv2.cvtColor(self.img_procesada, cv2.COLOR_BGR2HSV)
        # Ejemplo: segmentar color amarillo
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(self.img_procesada, self.img_procesada, mask=mask)
        self.img_procesada = result
        self.mostrar_imagen(self.label_procesada, self.img_procesada)

    def detectar_zonas_planas(self):
        if self.img_procesada is None:
            return
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(self.img_procesada, cv2.COLOR_BGR2GRAY)
        
        # Calcular gradiente con Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitud del gradiente
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # Normalizar y umbralizar para detectar zonas planas
        grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
        _, mask = cv2.threshold(grad_mag, 15, 255, cv2.THRESH_BINARY_INV)  # umbral bajo → zonas planas
        
        mask = mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask)
        num_zonas = num_labels - 1  # restamos 1 porque el label 0 es el fondo
        print(f"Zonas planas detectadas: {num_zonas}")
            
        # Resaltar zonas planas en color rojo
        resultado = self.img_original.copy()
        resultado[mask == 255] = [0, 0, 255]
        
        # Guardar y mostrar
        self.imagen_procesada = resultado
        self.mostrar_imagen(self.label_procesada, resultado)
    def detectar_zonas_planas(self):
        if self.img_procesada is None:
            return
        num_segments = 100  # Ajusta según la imagen
        segments = slic(self.img_procesada, n_segments=num_segments, compactness=10, sigma=1)
        homogenized = np.zeros_like(self.img_procesada)
        for seg_id in np.unique(segments):
            mask = segments == seg_id
            mean_color = np.mean(image[mask], axis=0)
            homogenized[mask] = mean_color
        total_zonas = len(np.unique(segments))
        print(f"Total de zonas planas en la imagen: {total_zonas}")
  
            
    def aplicar_gradiente_interno(self):
        if self.img_procesada is None:
            return
        k = self.kernel_size.value()
        kernel = np.ones((k, k), np.uint8)
        erosionada = cv2.erode(self.img_procesada, kernel, iterations=1)
        gradiente = cv2.subtract(self.img_procesada, erosionada)
        self.img_procesada = gradiente
        self.mostrar_imagen(self.label_procesada, self.img_procesada)

    def aplicar_gradiente_externo(self):
        if self.img_procesada is None:
            return
        k = self.kernel_size.value()
        kernel = np.ones((k, k), np.uint8)
        dilatada = cv2.dilate(self.img_procesada, kernel, iterations=1)
        gradiente = cv2.subtract(dilatada, self.img_procesada)
        self.img_procesada = gradiente
        self.mostrar_imagen(self.label_procesada, self.img_procesada)

    def aplicar_gradiente_total(self):
        if self.img_procesada is None:
            return
        k = self.kernel_size.value()
        kernel = np.ones((k, k), np.uint8)
        dilatada = cv2.dilate(self.img_procesada, kernel, iterations=1)
        erosionada = cv2.erode(self.img_procesada, kernel, iterations=1)
        gradiente = cv2.subtract(dilatada, erosionada)
        self.img_procesada = gradiente
        self.mostrar_imagen(self.label_procesada, self.img_procesada)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ImageProcessor()
    win.show()
    sys.exit(app.exec_())