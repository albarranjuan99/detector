from PIL import Image
import os

# Ruta de la carpeta que contiene las imágenes PNG
carpeta_imagenes = "D:\det\data\custom\images"

# Nueva resolución deseada
nueva_resolucion = (500, 500)  # Cambia a la resolución que desees

def cambiar_resolucion(imagen, nueva_resolucion):
    imagen = imagen.resize(nueva_resolucion, Image.ANTIALIAS)
    return imagen

def cambiar_resolucion_en_carpeta(carpeta, nueva_resolucion):
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".png"):
            ruta_completa = os.path.join(carpeta, archivo)
            imagen = Image.open(ruta_completa)
            imagen = cambiar_resolucion(imagen, nueva_resolucion)
            imagen.save(ruta_completa)

cambiar_resolucion_en_carpeta(carpeta_imagenes, nueva_resolucion)