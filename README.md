# 🧪 Simulador 3D de Reactor de Lecho Empacado

Simulador interactivo en 3D para reactores de lecho empacado con visualización avanzada, animación de flujo y análisis multiescala.

![Reactor 3D](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly)

## 🚀 Demo en Vivo

**[Abrir Simulador](https://share.streamlit.io)** ← Click aquí para usar el simulador

## ✨ Características

### Visualización 3D Avanzada
- 🎨 Renderizado 3D interactivo con Plotly
- 🌊 Animación de flujo con partículas trazadoras
- 🎯 Isosuperficies de concentración
- 🔄 Rotación automática 360°
- 📊 Colores según presión (escala Viridis)

### Modelado Multiescala
- ⚗️ Ecuación de Ergun (caída de presión)
- 🔥 Cinética de Arrhenius
- 📈 Balances de masa y energía
- 🧮 Análisis de Thiele (factor de efectividad)
- 🌡️ Perfiles axiales de temperatura, presión y conversión

### Escalas Industriales
- 🏭 **Industrial**: Reactores de 5-10 m de diámetro
- 🔬 **Piloto**: Validación de modelos (DN50)
- ⚗️ **Laboratorio**: Estudios cinéticos

### Instrumentación Realista
- 📏 Manómetros de presión
- 🌡️ Termopozos
- 🔩 Bridas y soportes estructurales
- 📐 Dimensiones reales según escala

## 🎮 Cómo Usar

1. **Selecciona la escala** del reactor (Industrial/Piloto/Laboratorio)
2. **Ajusta los parámetros** en el sidebar:
   - Temperatura de entrada
   - Presión de entrada
   - Velocidad superficial
   - Concentración inicial
   - Diámetro y altura del reactor
3. **Visualiza en 3D**:
   - Click en "▶ Rotar + Flujo" para iniciar animación
   - Hover sobre partículas para ver datos locales
   - Arrastra para rotar manualmente
4. **Analiza resultados**:
   - Perfiles axiales
   - Métricas clave
   - Análisis multiescala

## 📦 Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/reactor-simulator.git
cd reactor-simulator

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar simulador
streamlit run reactor_simulator.py
```

## 🛠️ Tecnologías

- **Streamlit**: Framework de aplicaciones web
- **Plotly**: Visualización 3D interactiva
- **NumPy**: Cálculos numéricos
- **SciPy**: Integración de ecuaciones diferenciales
- **Pandas**: Manejo de datos

## 📊 Ecuaciones Implementadas

### Ecuación de Ergun
```
ΔP/L = (150μv_s(1-ε)²)/(ε³d_p²) + (1.75ρv_s²(1-ε))/(ε³d_p)
```

### Cinética de Arrhenius
```
k = k₀ exp(-E_a/RT)
```

### Balance de Energía
```
ρ_f C_p v_s dT/dz = (-ΔH_r) r_A
```

## 🎯 Casos de Uso

- 📚 **Educación**: Enseñanza de ingeniería de reactores
- 🔬 **Investigación**: Análisis de fenómenos de transporte
- 🏭 **Industria**: Diseño preliminar de reactores
- 📊 **Presentaciones**: Visualización de conceptos complejos

## 📝 Documentación

- [Guía de Despliegue](DESPLEGAR_AHORA.md)
- [Guía de Uso](EJECUTAR_ANIMACION.md)
- [Documentación Completa](GUIA_EXPLICACION_COMPLETA.md)

## 🤝 Contribuir

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 👨‍💻 Autor

Desarrollado con ❤️ para la comunidad de ingeniería química

## 🌟 Agradecimientos

- Comunidad de Streamlit
- Plotly por las increíbles visualizaciones 3D
- Todos los que contribuyen al proyecto

---

⭐ Si te gusta este proyecto, dale una estrella en GitHub!

📧 ¿Preguntas? Abre un issue en el repositorio.
