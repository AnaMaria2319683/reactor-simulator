import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import odeint
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Simulador Reactor Lecho Empacado 3D", 
    layout="wide", 
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

# CSS para usar toda la página sin márgenes
st.markdown("""
    <style>
    /* Reducir padding para mejor visualización */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
        max-width: 100%;
    }
    
    /* Hacer que el contenido use todo el ancho */
    .stApp {
        margin: 0;
        padding: 0;
    }
    
    /* Ajustar el sidebar */
    section[data-testid="stSidebar"] {
        width: 280px !important;
    }
    
    /* Eliminar espacios extra */
    .element-container {
        margin: 0;
    }
    
    /* Hacer gráficos más grandes */
    .js-plotly-plot {
        width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("Simulador de Reactor de Lecho Empacado 3D")
st.markdown("""
Este simulador integra modelado multiescala, morfología 3D y análisis de transporte para reactores de lecho empacado.
Basado en principios de CFD, cinética química y fenómenos de transporte.
""")

# Sidebar con parámetros
st.sidebar.header("Parametros del Reactor")

# Parámetros geométricos
st.sidebar.subheader("Geometría del Reactor")

# Selector de escala
escala = st.sidebar.selectbox(
    "Escala del reactor",
    ["Industrial", "Piloto", "Laboratorio", "Personalizado"],
    index=0
)

# Configurar dimensiones según escala
if escala == "Industrial":
    D_tube = st.sidebar.slider("Diámetro del tubo (m)", 5.0, 10.0, 7.0, 0.5)
    L_reactor = st.sidebar.slider("Longitud del reactor (m)", 5.0, 15.0, 11.0, 0.5)
    d_particle = st.sidebar.slider("Diámetro de partícula (mm)", 1.0, 13.0, 3.0, 0.5) / 1000
    porosity = st.sidebar.slider("Porosidad (ε)", 0.35, 0.50, 0.40, 0.01)
    st.sidebar.info("Escala Industrial: Reactores de gran volumen (5-10 m diametro)")
elif escala == "Piloto":
    D_tube = st.sidebar.slider("Diámetro del tubo (m)", 0.01, 0.10, 0.05, 0.01)
    L_reactor = st.sidebar.slider("Longitud del reactor (m)", 0.5, 2.0, 1.0, 0.1)
    d_particle = st.sidebar.slider("Diámetro de partícula (mm)", 1.0, 10.0, 2.0, 0.5) / 1000
    porosity = st.sidebar.slider("Porosidad (ε)", 0.35, 0.50, 0.40, 0.01)
    st.sidebar.info("Escala Piloto: Validacion de modelos (DN50)")
elif escala == "Laboratorio":
    D_tube = st.sidebar.slider("Diámetro del tubo (m)", 0.01, 0.05, 0.025, 0.005)
    L_reactor = st.sidebar.slider("Longitud del reactor (m)", 0.1, 1.0, 0.5, 0.1)
    d_particle = st.sidebar.slider("Diámetro de partícula (mm)", 0.3, 5.0, 1.0, 0.1) / 1000
    porosity = st.sidebar.slider("Porosidad (ε)", 0.35, 0.50, 0.40, 0.01)
    st.sidebar.info("Escala Laboratorio: Estudios cineticos")
else:  # Personalizado
    D_tube = st.sidebar.slider("Diámetro del tubo (m)", 0.01, 10.0, 0.1, 0.01)
    L_reactor = st.sidebar.slider("Longitud del reactor (m)", 0.1, 15.0, 2.0, 0.1)
    d_particle = st.sidebar.slider("Diámetro de partícula (mm)", 0.3, 25.0, 5.0, 0.1) / 1000
    porosity = st.sidebar.slider("Porosidad (ε)", 0.30, 0.60, 0.40, 0.01)

# Calcular y mostrar relación N
N_ratio = D_tube / d_particle
st.sidebar.markdown("---")
st.sidebar.markdown("### 📐 Relaciones Críticas")
st.sidebar.metric("N = D/dp", f"{N_ratio:.1f}")
if N_ratio < 10:
    st.sidebar.warning("N < 10: Efecto de pared significativo")
else:
    st.sidebar.success("N >= 10: Efecto de pared minimo")

st.sidebar.metric("L/D", f"{L_reactor/D_tube:.1f}")
st.sidebar.metric("Volumen", f"{np.pi*(D_tube/2)**2*L_reactor:.2f} m³")

# Parámetros operacionales
st.sidebar.subheader("Condiciones Operacionales")
T_inlet = st.sidebar.slider("Temperatura entrada (°C)", 100, 600, 300, 10)
P_inlet = st.sidebar.slider("Presión entrada (bar)", 1, 50, 10, 1)
v_superficial = st.sidebar.slider("Velocidad superficial (m/s)", 0.01, 2.0, 0.5, 0.01)
C_inlet = st.sidebar.slider("Concentración entrada (mol/m³)", 10, 1000, 100, 10)

# Parámetros de reacción
st.sidebar.subheader("Cinética Química")
k0 = st.sidebar.number_input("Factor pre-exponencial k₀ (1/s)", value=1e6, format="%.2e")
Ea = st.sidebar.slider("Energía de activación (kJ/mol)", 20, 200, 80, 5)
reaction_order = st.sidebar.selectbox("Orden de reacción", [1, 2], index=0)

# Cálculos del modelo
def calculate_ergun_pressure_drop(L, d_p, epsilon, v_s, rho=1.2, mu=1.8e-5):
    """Ecuación de Ergun para caída de presión"""
    term1 = 150 * mu * v_s * (1 - epsilon)**2 / (epsilon**3 * d_p**2)
    term2 = 1.75 * rho * v_s**2 * (1 - epsilon) / (epsilon**3 * d_p)
    dP_dL = term1 + term2
    return dP_dL * L / 1e5  # Convertir a bar

def arrhenius_rate(T, k0, Ea):
    """Constante de velocidad de Arrhenius"""
    R = 8.314  # J/(mol·K)
    return k0 * np.exp(-Ea * 1000 / (R * T))

def reactor_model(y, z, k, order, v, epsilon):
    """Sistema de EDOs para el reactor"""
    C = y[0]
    T = y[1]
    
    # Tasa de reacción
    if order == 1:
        r = -k * C
    else:
        r = -k * C**2
    
    # Balance de masa
    dC_dz = r * (1 - epsilon) / (v * epsilon)
    
    # Balance de energía simplificado (adiabático)
    dT_dz = -r * 50000 / (rho_fluid * Cp_fluid * v * epsilon)  # ΔH_rxn = 50 kJ/mol
    
    return [dC_dz, dT_dz]

# Constantes físicas
rho_fluid = 1.2  # kg/m³
Cp_fluid = 1000  # J/(kg·K)

# Resolución del modelo
z_points = np.linspace(0, L_reactor, 100)
T_K = T_inlet + 273.15
k_reaction = arrhenius_rate(T_K, k0, Ea)

# Condiciones iniciales
y0 = [C_inlet, T_K]

# Resolver EDOs
solution = odeint(reactor_model, y0, z_points, 
                  args=(k_reaction, reaction_order, v_superficial, porosity))

C_profile = solution[:, 0]
T_profile = solution[:, 1] - 273.15  # Convertir a °C

# Cálculo de conversión
conversion = (1 - C_profile / C_inlet) * 100

# Caída de presión
delta_P = calculate_ergun_pressure_drop(L_reactor, d_particle, porosity, 
                                        v_superficial, rho_fluid)

# Layout de columnas - usar más espacio para visualizaciones
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Perfiles Axiales del Reactor")
    
    # Crear figura con subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Concentración vs Posición', 'Temperatura vs Posición',
                       'Conversión vs Posición', 'Perfil de Presión'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Concentración
    fig.add_trace(
        go.Scatter(x=z_points, y=C_profile, name='Concentración',
                  line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Temperatura
    fig.add_trace(
        go.Scatter(x=z_points, y=T_profile, name='Temperatura',
                  line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    # Conversión
    fig.add_trace(
        go.Scatter(x=z_points, y=conversion, name='Conversión',
                  line=dict(color='green', width=3)),
        row=2, col=1
    )
    
    # Presión
    P_profile = P_inlet - (delta_P * z_points / L_reactor)
    fig.add_trace(
        go.Scatter(x=z_points, y=P_profile, name='Presión',
                  line=dict(color='purple', width=3)),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Posición axial (m)", row=1, col=1)
    fig.update_xaxes(title_text="Posición axial (m)", row=1, col=2)
    fig.update_xaxes(title_text="Posición axial (m)", row=2, col=1)
    fig.update_xaxes(title_text="Posición axial (m)", row=2, col=2)
    
    fig.update_yaxes(title_text="C (mol/m³)", row=1, col=1)
    fig.update_yaxes(title_text="T (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Conversión (%)", row=2, col=1)
    fig.update_yaxes(title_text="P (bar)", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Resultados Clave")
    
    # Métricas
    st.metric("Conversión Final", f"{conversion[-1]:.2f}%")
    st.metric("Caída de Presión", f"{delta_P:.3f} bar")
    st.metric("Temperatura Salida", f"{T_profile[-1]:.1f} °C")
    st.metric("Relación D/dp", f"{D_tube/d_particle:.1f}")
    
    # Información adicional
    st.info(f"""
    **Régimen de Flujo:**
    - Re partícula: {(rho_fluid * v_superficial * d_particle / 1.8e-5):.0f}
    - Porosidad efectiva: {porosity:.3f}
    """)
    
    if D_tube/d_particle < 10:
        st.warning("Efecto de pared significativo (D/dp < 10)")

# Separador visual
st.markdown("---")

# Visualización 3D del lecho empacado
st.subheader("Visualizacion 3D del Reactor - Vista con Instrumentacion")
st.markdown("<br>", unsafe_allow_html=True)

# Controles de visualización
col_viz1, col_viz2, col_viz3 = st.columns(3)
with col_viz1:
    n_particles_viz = st.slider("Número de partículas", 20, 150, 80, 10)
with col_viz2:
    show_components = st.checkbox("Mostrar componentes estructurales", value=True)
with col_viz3:
    show_instruments = st.checkbox("Mostrar instrumentación", value=True)

st.markdown("<br>", unsafe_allow_html=True)

# Generar geometría 3D del lecho empacado con empaquetamiento aleatorio realista
@st.cache_data
def generate_realistic_packed_bed(D_tube, L_reactor, d_particle, n_particles=80, seed=42):
    """
    Genera empaquetamiento aleatorio realista con detección de colisiones
    Método Monte Carlo mejorado con sedimentación por gravedad
    """
    np.random.seed(seed)
    particles = []
    max_attempts = n_particles * 100
    attempts = 0
    
    # Función para verificar colisiones
    def check_collision(new_pos, existing_particles, min_distance):
        if len(existing_particles) == 0:
            return False
        distances = np.sqrt(np.sum((np.array(existing_particles) - new_pos)**2, axis=1))
        return np.any(distances < min_distance)
    
    # Generar partículas con sedimentación por gravedad
    while len(particles) < n_particles and attempts < max_attempts:
        # Posición horizontal aleatoria
        r = np.random.uniform(0, D_tube/2 - d_particle/2 * 1.1)
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Altura: intentar colocar lo más bajo posible (simulando gravedad)
        if len(particles) == 0:
            z = d_particle/2
        else:
            # Buscar altura mínima sin colisión
            z_test = d_particle/2
            found = False
            for z_candidate in np.linspace(d_particle/2, L_reactor - d_particle/2, 50):
                new_pos = np.array([x, y, z_candidate])
                if not check_collision(new_pos, particles, d_particle * 1.05):
                    z_test = z_candidate
                    found = True
                    break
            
            if not found:
                attempts += 1
                continue
            z = z_test
        
        new_particle = [x, y, z]
        
        # Verificar que está dentro del reactor
        if z < d_particle/2 or z > L_reactor - d_particle/2:
            attempts += 1
            continue
        
        particles.append(new_particle)
        attempts = 0  # Resetear intentos después de éxito
    
    particles_array = np.array(particles)
    
    # Calcular porosidad real
    V_reactor = np.pi * (D_tube/2)**2 * L_reactor
    V_particles = len(particles) * (4/3) * np.pi * (d_particle/2)**3
    actual_porosity = 1 - V_particles / V_reactor
    
    return particles_array, actual_porosity

# Generar partículas con empaquetamiento realista
particles_pos, actual_porosity = generate_realistic_packed_bed(
    D_tube, L_reactor, d_particle, n_particles=n_particles_viz
)

# Mostrar información del empaquetamiento en columnas
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Partículas generadas", len(particles_pos))
with col_info2:
    st.metric("Porosidad real", f"{actual_porosity:.3f}")
with col_info3:
    st.metric("Diferencia vs teórica", f"{(actual_porosity - porosity)*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════
# GENERACIÓN DE PARTÍCULAS DEL LECHO EMPACADO
# ═══════════════════════════════════════════════════════════════════
# Se generan partículas aleatorias dentro del reactor cilíndrico
# Cada partícula tiene una posición (x, y, z) y propiedades calculadas

np.random.seed(42)
particles_simple = []

# Generar posiciones aleatorias dentro del cilindro
for _ in range(n_particles_viz):
    r = np.random.uniform(0, D_tube/2 - d_particle)  # Radio desde el centro
    theta = np.random.uniform(0, 2*np.pi)  # Ángulo
    z = np.random.uniform(d_particle, L_reactor - d_particle)  # Altura
    
    # Convertir coordenadas cilíndricas a cartesianas
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    particles_simple.append([x, y, z])

particles_simple = np.array(particles_simple)

# ═══════════════════════════════════════════════════════════════════
# CÁLCULO DE PROPIEDADES EN CADA PARTÍCULA
# ═══════════════════════════════════════════════════════════════════
# Para cada partícula, se interpola temperatura, presión y conversión
# según su altura en el reactor

colors_temp = []
colors_pressure = []
particle_info = []

for pos in particles_simple:
    # Encontrar índice correspondiente a la altura de la partícula
    z_idx = int((pos[2] / L_reactor) * (len(T_profile) - 1))
    z_idx = max(0, min(z_idx, len(T_profile) - 1))
    
    # Interpolar propiedades
    temp = T_profile[z_idx]
    pressure = P_inlet - (delta_P * pos[2] / L_reactor)
    conv = conversion[z_idx]
    
    colors_temp.append(temp)
    colors_pressure.append(pressure)
    particle_info.append({'temp': temp, 'pressure': pressure, 'conv': conv, 'z': pos[2]})

# ═══════════════════════════════════════════════════════════════════
# VISUALIZACIÓN 3D: LECHO DE CATALIZADOR
# ═══════════════════════════════════════════════════════════════════
# Las partículas se muestran con color según presión (escala Viridis)
# Textura metálica simulada con bordes oscuros

# Inicializar figura 3D
fig_3d = go.Figure()

fig_3d.add_trace(go.Scatter3d(
    x=particles_simple[:, 0],
    y=particles_simple[:, 1],
    z=particles_simple[:, 2],
    mode='markers',
    marker=dict(
        size=10,
        color=colors_pressure,  # Color según PRESIÓN
        colorscale='Viridis',  # Escala metálica: violeta→verde→amarillo
        showscale=True,
        colorbar=dict(title='Presión (bar)', x=1.15, len=0.6, y=0.5, thickness=20, tickfont=dict(size=12)),
        line=dict(color='rgba(50, 50, 50, 0.8)', width=1.5),  # Borde oscuro = efecto metálico
        cmin=min(colors_pressure),
        cmax=max(colors_pressure),
        opacity=0.95
    ),
    name='Lecho de Catalizador',
    text=[f'<b>Partícula de catalizador</b><br>dp = {d_particle*1000:.2f} mm<br>' +
          f'Temperatura: {info["temp"]:.1f}°C<br>Presión: {info["pressure"]:.2f} bar<br>' +
          f'Conversión: {info["conv"]:.1f}%<br>Altura: {info["z"]:.3f} m'
          for info in particle_info],
    hovertemplate='%{text}<extra></extra>',
    customdata=[[info['temp'], info['pressure'], info['conv']] for info in particle_info]
))

# ═══════════════════════════════════════════════════════════════════
# ANIMACIÓN DE FLUJO: PARTÍCULAS TRAZADORAS
# ═══════════════════════════════════════════════════════════════════
# Se crean pequeñas esferas que se mueven desde z=0 hasta z=L
# simulando el flujo del fluido a través del lecho empacado

n_tracers = 15  # Número de partículas trazadoras
n_frames_flow = 50  # Frames de animación

# Generar trayectorias de trazadores (evitando partículas del lecho)
tracer_paths = []
for i in range(n_tracers):
    # Posición inicial aleatoria en la entrada
    r_start = np.random.uniform(0, D_tube/2 * 0.8)
    theta_start = np.random.uniform(0, 2*np.pi)
    x_start = r_start * np.cos(theta_start)
    y_start = r_start * np.sin(theta_start)
    
    # Crear trayectoria con pequeñas variaciones (simula flujo tortuoso)
    z_path = np.linspace(0, L_reactor, n_frames_flow)
    x_path = x_start + np.random.normal(0, D_tube/20, n_frames_flow).cumsum() * 0.01
    y_path = y_start + np.random.normal(0, D_tube/20, n_frames_flow).cumsum() * 0.01
    
    # Mantener dentro del reactor
    r_path = np.sqrt(x_path**2 + y_path**2)
    mask = r_path > D_tube/2 * 0.9
    x_path[mask] *= 0.8
    y_path[mask] *= 0.8
    
    tracer_paths.append((x_path, y_path, z_path))

# Añadir trazadores iniciales (frame 0)
for i, (x_path, y_path, z_path) in enumerate(tracer_paths):
    fig_3d.add_trace(go.Scatter3d(
        x=[x_path[0]],
        y=[y_path[0]],
        z=[z_path[0]],
        mode='markers',
        marker=dict(size=6, color='cyan', opacity=0.8, symbol='circle'),
        name=f'Trazador {i+1}' if i == 0 else '',
        showlegend=(i == 0),
        hovertemplate=f'<b>Partícula de fluido</b><br>Velocidad: {v_superficial:.3f} m/s<extra></extra>'
    ))

# AGREGAR ISOSUPERFICIE DE CONCENTRACIÓN (Gradiente 3D)
# Crear grid 3D para la concentración
n_grid = 30  # Resolución del grid
x_grid = np.linspace(-D_tube/2, D_tube/2, n_grid)
y_grid = np.linspace(-D_tube/2, D_tube/2, n_grid)
z_grid_iso = np.linspace(0, L_reactor, n_grid)

X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid_iso)

# Calcular concentración en cada punto del grid
C_grid = np.zeros_like(X)
for i in range(n_grid):
    for j in range(n_grid):
        for k in range(n_grid):
            x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
            r = np.sqrt(x**2 + y**2)
            
            # Solo dentro del reactor
            if r <= D_tube/2:
                # Interpolar concentración según altura
                z_idx = int((z / L_reactor) * (len(C_profile) - 1))
                z_idx = max(0, min(z_idx, len(C_profile) - 1))
                C_grid[i,j,k] = C_profile[z_idx]
            else:
                C_grid[i,j,k] = np.nan

# Añadir isosuperficie de concentración
fig_3d.add_trace(go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=C_grid.flatten(),
    isomin=C_inlet * 0.3,
    isomax=C_inlet * 0.9,
    surface_count=5,  # Número de superficies
    colorscale='Plasma',
    opacity=0.15,
    caps=dict(x_show=False, y_show=False, z_show=False),
    showscale=True,
    colorbar=dict(
        title='Concentración<br>(mol/m³)',
        x=1.0,
        len=0.4,
        y=0.2,
        thickness=15,
        tickfont=dict(size=10, color='white')
    ),
    name='Gradiente de Concentración',
    hovertemplate='<b>Concentración</b><br>C: %{value:.1f} mol/m³<extra></extra>'
))

if show_components:
    # Grosor de pared según escala
    if escala == "Industrial":
        wall_thickness = 0.05  # 50 mm
    elif escala == "Piloto":
        wall_thickness = 0.005  # 5 mm
    else:
        wall_thickness = 0.003  # 3 mm
    
    D_exterior = D_tube + 2 * wall_thickness
    
    # Pared externa (gris oscuro)
    theta_cyl = np.linspace(0, 2*np.pi, 60)
    z_cyl = np.linspace(0, L_reactor, 50)
    theta_grid, z_grid = np.meshgrid(theta_cyl, z_cyl)
    
    x_ext = (D_exterior/2) * np.cos(theta_grid)
    y_ext = (D_exterior/2) * np.sin(theta_grid)
    
    fig_3d.add_trace(go.Surface(
        x=x_ext, y=y_ext, z=z_grid,
        colorscale=[[0, 'darkgray'], [1, 'gray']],
        showscale=False,
        opacity=0.5,
        name='Carcasa Externa',
        hovertemplate='<b>Carcasa Externa</b><br>Espesor: ' + f'{wall_thickness*1000:.0f} mm<extra></extra>'
    ))
    
    # Pared interna (más clara y transparente)
    x_int = (D_tube/2) * np.cos(theta_grid)
    y_int = (D_tube/2) * np.sin(theta_grid)
    
    fig_3d.add_trace(go.Surface(
        x=x_int, y=y_int, z=z_grid,
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        showscale=False,
        opacity=0.08,
        name='Pared Interna',
        hoverinfo='skip'
    ))
    
    # Bridas
    D_brida = D_exterior * 1.3
    theta_brida = np.linspace(0, 2*np.pi, 60)
    r_brida = np.linspace(D_exterior/2, D_brida/2, 15)
    theta_b_grid, r_b_grid = np.meshgrid(theta_brida, r_brida)
    
    x_brida = r_b_grid * np.cos(theta_b_grid)
    y_brida = r_b_grid * np.sin(theta_b_grid)
    
    # Brida inferior
    z_brida_inf = np.ones_like(x_brida) * 0
    fig_3d.add_trace(go.Surface(
        x=x_brida, y=y_brida, z=z_brida_inf,
        colorscale=[[0, 'dimgray'], [1, 'dimgray']],
        showscale=False,
        opacity=0.9,
        name='Brida Inferior',
        hovertemplate='<b>Brida Inferior</b><extra></extra>'
    ))
    
    # Brida superior
    z_brida_sup = np.ones_like(x_brida) * L_reactor
    fig_3d.add_trace(go.Surface(
        x=x_brida, y=y_brida, z=z_brida_sup,
        colorscale=[[0, 'dimgray'], [1, 'dimgray']],
        showscale=False,
        opacity=0.9,
        name='Brida Superior',
        hovertemplate='<b>Brida Superior</b><extra></extra>'
    ))
    
    # Tuberías
    D_tuberia = D_tube * 0.4
    L_tuberia = L_reactor * 0.15
    
    # Tubería de entrada (azul)
    z_entrada = np.linspace(-L_tuberia, 0, 20)
    theta_tub = np.linspace(0, 2*np.pi, 40)
    theta_t_grid, z_t_grid = np.meshgrid(theta_tub, z_entrada)
    
    x_tub_ent = (D_tuberia/2) * np.cos(theta_t_grid)
    y_tub_ent = (D_tuberia/2) * np.sin(theta_t_grid)
    
    fig_3d.add_trace(go.Surface(
        x=x_tub_ent, y=y_tub_ent, z=z_t_grid,
        colorscale=[[0, 'steelblue'], [1, 'steelblue']],
        showscale=False,
        opacity=0.8,
        name='Tubería Entrada',
        hovertemplate='<b>Tubería de Entrada</b><extra></extra>'
    ))
    
    # Tubería de salida (coral)
    z_salida = np.linspace(L_reactor, L_reactor + L_tuberia, 20)
    theta_t_grid_sal, z_t_grid_sal = np.meshgrid(theta_tub, z_salida)
    
    x_tub_sal = (D_tuberia/2) * np.cos(theta_t_grid_sal)
    y_tub_sal = (D_tuberia/2) * np.sin(theta_t_grid_sal)
    
    fig_3d.add_trace(go.Surface(
        x=x_tub_sal, y=y_tub_sal, z=z_t_grid_sal,
        colorscale=[[0, 'coral'], [1, 'coral']],
        showscale=False,
        opacity=0.8,
        name='Tubería Salida',
        hovertemplate='<b>Tubería de Salida</b><extra></extra>'
    ))
    
    # Soportes estructurales
    n_soportes = 4
    for i in range(n_soportes):
        angle = 2 * np.pi * i / n_soportes
        x_soporte = (D_exterior/2 + D_exterior*0.1) * np.cos(angle)
        y_soporte = (D_exterior/2 + D_exterior*0.1) * np.sin(angle)
        
        fig_3d.add_trace(go.Scatter3d(
            x=[x_soporte, x_soporte],
            y=[y_soporte, y_soporte],
            z=[-L_tuberia, L_reactor],
            mode='lines',
            line=dict(color='dimgray', width=10),
            showlegend=False,
            hoverinfo='skip'
        ))
else:
    # Solo pared simple si no se muestran componentes
    theta_cyl = np.linspace(0, 2*np.pi, 50)
    z_cyl = np.linspace(0, L_reactor, 50)
    theta_grid, z_grid = np.meshgrid(theta_cyl, z_cyl)
    x_cyl = (D_tube/2) * np.cos(theta_grid)
    y_cyl = (D_tube/2) * np.sin(theta_grid)
    
    P_cyl = P_inlet - (delta_P * z_grid / L_reactor)
    
    fig_3d.add_trace(go.Surface(
        x=x_cyl, 
        y=y_cyl, 
        z=z_grid,
        opacity=0.08,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        showscale=False,
        surfacecolor=P_cyl,
        hovertemplate=(
            '<b>Pared del Reactor</b><br>' +
            'Altura: %{z:.3f} m<br>' +
            'Presión: %{surfacecolor:.2f} bar<br>' +
            '<extra></extra>'
        ),
        name='Pared'
    ))

# Añadir instrumentación si está activada
if show_instruments:
    # Función para obtener temperatura en posición z
    def get_temperature_at_z(z_pos):
        idx = int((z_pos / L_reactor) * (len(T_profile) - 1))
        idx = max(0, min(idx, len(T_profile) - 1))
        return T_profile[idx]
    
    # Función para obtener presión en posición z
    def get_pressure_at_z(z_pos):
        return P_inlet - (delta_P * z_pos / L_reactor)
    
    # MANÓMETROS (Pressure Gauges) - Amarillo
    if show_components:
        D_ref = D_exterior
    else:
        D_ref = D_tube
    
    manometro_positions = [
        (D_ref/2 * 1.3, 0, L_reactor * 0.9, 'Manómetro Superior'),
        (D_ref/2 * 1.3, 0, L_reactor * 0.5, 'Manómetro Medio'),
        (D_ref/2 * 1.3, 0, L_reactor * 0.1, 'Manómetro Inferior')
    ]
    
    for x, y, z, name in manometro_positions:
        # Cuerpo del manómetro
        fig_3d.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(
                size=15,
                color='yellow',
                symbol='circle',
                line=dict(color='black', width=2)
            ),
            name=name,
            hovertemplate=f'<b>{name}</b><br>Altura: {z:.2f} m<br>Presión: {get_pressure_at_z(z):.2f} bar<extra></extra>'
        ))
        
        # Línea de conexión
        fig_3d.add_trace(go.Scatter3d(
            x=[D_ref/2, x],
            y=[0, y],
            z=[z, z],
            mode='lines',
            line=dict(color='gray', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # TERMOPOZOS (Thermowells) - Rojo
    termopozo_positions = [
        (0, D_ref/2 * 1.3, L_reactor * 0.8, 'Termopozo Superior'),
        (0, D_ref/2 * 1.3, L_reactor * 0.3, 'Termopozo Inferior')
    ]
    
    for x, y, z, name in termopozo_positions:
        fig_3d.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='diamond',
                line=dict(color='darkred', width=2)
            ),
            name=name,
            hovertemplate=f'<b>{name}</b><br>Altura: {z:.2f} m<br>Temperatura: {get_temperature_at_z(z):.1f}°C<extra></extra>'
        ))
        
        # Línea de conexión
        fig_3d.add_trace(go.Scatter3d(
            x=[0, x],
            y=[D_ref/2, y],
            z=[z, z],
            mode='lines',
            line=dict(color='darkred', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Flechas de flujo
    if show_components:
        # Entrada (azul)
        fig_3d.add_trace(go.Cone(
            x=[0], y=[0], z=[-L_tuberia/2],
            u=[0], v=[0], w=[L_tuberia/2],
            colorscale=[[0, 'blue'], [1, 'blue']],
            showscale=False,
            sizemode='absolute',
            sizeref=L_tuberia/3,
            name='Entrada de Flujo',
            hovertemplate=f'<b>Entrada de Flujo</b><br>T: {T_inlet:.1f}°C<br>P: {P_inlet:.1f} bar<extra></extra>'
        ))
        
        # Salida (rojo/naranja)
        fig_3d.add_trace(go.Cone(
            x=[0], y=[0], z=[L_reactor + L_tuberia/2],
            u=[0], v=[0], w=[L_tuberia/2],
            colorscale=[[0, 'orangered'], [1, 'orangered']],
            showscale=False,
            sizemode='absolute',
            sizeref=L_tuberia/3,
            name='Salida de Flujo',
            hovertemplate=f'<b>Salida de Flujo</b><br>T: {T_profile[-1]:.1f}°C<br>P: {P_profile[-1]:.1f} bar<extra></extra>'
        ))
    else:
        # Flechas simples
        fig_3d.add_trace(go.Cone(
            x=[0], y=[0], z=[L_reactor * 0.05],
            u=[0], v=[0], w=[L_reactor * 0.1],
            colorscale=[[0, 'blue'], [1, 'blue']],
            showscale=False,
            sizemode='absolute',
            sizeref=L_reactor * 0.08,
            name='Entrada de Flujo',
            hovertemplate=f'<b>Entrada de Flujo</b><br>T: {T_inlet:.1f}°C<br>P: {P_inlet:.1f} bar<extra></extra>'
        ))
        
        fig_3d.add_trace(go.Cone(
            x=[0], y=[0], z=[L_reactor * 0.95],
            u=[0], v=[0], w=[L_reactor * 0.1],
            colorscale=[[0, 'orangered'], [1, 'orangered']],
            showscale=False,
            sizemode='absolute',
            sizeref=L_reactor * 0.08,
            name='Salida de Flujo',
            hovertemplate=f'<b>Salida de Flujo</b><br>T: {T_profile[-1]:.1f}°C<br>P: {P_profile[-1]:.1f} bar<extra></extra>'
        ))

# Configurar layout estilo industrial con fondo oscuro
fig_3d.update_layout(
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Altura (m)',
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=0.8),
            center=dict(x=0, y=0, z=0.5),
            up=dict(x=0, y=0, z=1)
        ),
        # Fondo gris oscuro para resaltar colores
        xaxis=dict(
            backgroundcolor="rgb(40, 40, 45)",
            gridcolor="rgb(80, 80, 85)",
            showbackground=True,
            zerolinecolor="rgb(100, 100, 105)",
            title=dict(text='X (m)', font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            backgroundcolor="rgb(40, 40, 45)",
            gridcolor="rgb(80, 80, 85)",
            showbackground=True,
            zerolinecolor="rgb(100, 100, 105)",
            title=dict(text='Y (m)', font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        zaxis=dict(
            backgroundcolor="rgb(40, 40, 45)",
            gridcolor="rgb(80, 80, 85)",
            showbackground=True,
            zerolinecolor="rgb(100, 100, 105)",
            title=dict(text='Altura (m)', font=dict(color='white')),
            tickfont=dict(color='white')
        ),
        # Iluminación tipo estudio
        camera_projection=dict(type='perspective')
    ),
    height=950,
    title=dict(
        text='<b>Vista 3D con Instrumentación y Dimensiones Reales</b>',
        x=0.5,
        xanchor='center',
        font=dict(size=16, color='white'),
        y=0.98,
        yanchor='top'
    ),
    paper_bgcolor='rgb(30, 30, 35)',  # Fondo oscuro del papel
    plot_bgcolor='rgb(30, 30, 35)',   # Fondo oscuro del plot
    showlegend=True,
    legend=dict(
        x=0.02,
        y=0.95,
        bgcolor='rgba(50, 50, 55, 0.9)',
        bordercolor='white',
        borderwidth=2,
        font=dict(color='white', size=10)
    ),
    hovermode='closest',
    margin=dict(t=60, b=20, l=20, r=20),
    updatemenus=[
        dict(
            type="buttons",
            showactive=True,
            buttons=[
                dict(
                    label="▶ Rotar + Flujo",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 100, "redraw": True},  # Más lento para ver flujo
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }]
                ),
                dict(
                    label="⏸ Pausar",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                ),
                dict(
                    label="🔄 Reset",
                    method="relayout",
                    args=[{
                        "scene.camera.eye": {"x": 1.8, "y": 1.8, "z": 0.8},
                        "scene.camera.center": {"x": 0, "y": 0, "z": 0.5}
                    }]
                )
            ],
            x=0.02,
            y=0.02,
            bgcolor='rgba(50, 50, 55, 0.9)',
            bordercolor='white',
            font=dict(color='white')
        )
    ]
)

# Crear frames para rotación Y animación de flujo
frames_3d = []
for i in range(max(36, n_frames_flow)):  # Usar el máximo entre rotación y flujo
    angle = 2 * np.pi * i / 36
    
    # Frame de rotación de cámara
    frame_data = []
    
    # Actualizar posición de trazadores (animación de flujo)
    if i < n_frames_flow:
        for tracer_idx, (x_path, y_path, z_path) in enumerate(tracer_paths):
            frame_data.append(go.Scatter3d(
                x=[x_path[i]],
                y=[y_path[i]],
                z=[z_path[i]],
                mode='markers',
                marker=dict(size=6, color='cyan', opacity=0.8),
                showlegend=False,
                hovertemplate=f'<b>Partícula de fluido</b><br>Altura: {z_path[i]:.3f} m<extra></extra>'
            ))
    
    frames_3d.append(go.Frame(
        data=frame_data if frame_data else None,
        layout=dict(scene=dict(camera=dict(
            eye=dict(x=1.8*np.cos(angle), y=1.8*np.sin(angle), z=0.8)
        ))),
        name=f'frame{i}'
    ))

fig_3d.frames = frames_3d

st.plotly_chart(fig_3d, use_container_width=True)

# Separador visual
st.markdown("<br><br>", unsafe_allow_html=True)

# Resumen de datos
st.markdown("""
<div style='background-color: #2a2a2f; color: white; padding: 20px; border-radius: 10px; border: 2px solid #555; margin-bottom: 20px;'>
    <h3 style='margin-top: 0; color: #4CAF50; text-align: center;'>📊 Resumen de Datos del Reactor</h3>
</div>
""", unsafe_allow_html=True)

col_summary1, col_summary2 = st.columns([1, 1])

with col_summary1:
    st.markdown("""
    <div style='background-color: #2a2a2f; color: white; padding: 20px; border-radius: 10px; border: 2px solid #555;'>
        <h4 style='color: #FFA726; margin-top: 0; margin-bottom: 15px;'>⚙️ Parámetros Operacionales</h4>
        <table style='width: 100%; font-size: 14px;'>
            <tr><td><b>Temperatura entrada:</b></td><td style='text-align: right;'>{:.1f}°C</td></tr>
            <tr><td><b>Temperatura salida:</b></td><td style='text-align: right; color: #FF6B6B;'>{:.1f}°C</td></tr>
            <tr><td><b>ΔT (cambio):</b></td><td style='text-align: right; color: #4ECDC4;'>{:.1f}°C</td></tr>
            <tr><td colspan='2'><hr style='border-color: #555;'></td></tr>
            <tr><td><b>Presión entrada:</b></td><td style='text-align: right;'>{:.2f} bar</td></tr>
            <tr><td><b>Presión salida:</b></td><td style='text-align: right; color: #FF6B6B;'>{:.2f} bar</td></tr>
            <tr><td><b>ΔP (caída):</b></td><td style='text-align: right; color: #FFD93D;'>{:.3f} bar</td></tr>
            <tr><td colspan='2'><hr style='border-color: #555;'></td></tr>
            <tr><td><b>Conversión final:</b></td><td style='text-align: right; color: #6BCF7F;'>{:.2f}%</td></tr>
            <tr><td><b>Velocidad superficial:</b></td><td style='text-align: right;'>{:.3f} m/s</td></tr>
        </table>
    </div>
    """.format(
        T_inlet, T_profile[-1], T_profile[-1] - T_inlet,
        P_inlet, P_profile[-1], delta_P,
        conversion[-1], v_superficial
    ), unsafe_allow_html=True)

with col_summary2:
    st.markdown("""
    <div style='background-color: #2a2a2f; color: white; padding: 20px; border-radius: 10px; border: 2px solid #555;'>
        <h4 style='color: #FFA726; margin-top: 0; margin-bottom: 15px;'>📐 Geometría del Reactor</h4>
        <table style='width: 100%; font-size: 14px;'>
            <tr><td><b>Diámetro (D):</b></td><td style='text-align: right;'>{:.3f} m</td></tr>
            <tr><td><b>Altura (L):</b></td><td style='text-align: right;'>{:.3f} m</td></tr>
            <tr><td><b>Relación L/D:</b></td><td style='text-align: right;'>{:.2f}</td></tr>
            <tr><td colspan='2'><hr style='border-color: #555;'></td></tr>
            <tr><td><b>Diámetro partícula:</b></td><td style='text-align: right;'>{:.2f} mm</td></tr>
            <tr><td><b>Relación N (D/dp):</b></td><td style='text-align: right; color: {};'>{:.0f} {}</td></tr>
            <tr><td colspan='2'><hr style='border-color: #555;'></td></tr>
            <tr><td><b>Volumen del lecho:</b></td><td style='text-align: right;'>{:.2f} m³</td></tr>
            <tr><td><b>Partículas mostradas:</b></td><td style='text-align: right;'>{}</td></tr>
            <tr><td><b>Porosidad:</b></td><td style='text-align: right;'>{:.3f}</td></tr>
        </table>
    </div>
    """.format(
        D_tube, L_reactor, L_reactor/D_tube,
        d_particle*1000, 
        '#6BCF7F' if N_ratio >= 10 else '#FFD93D', N_ratio, '✓' if N_ratio >= 10 else '⚠',
        np.pi*(D_tube/2)**2*L_reactor,
        len(particles_simple),
        porosity
    ), unsafe_allow_html=True)

# Analisis rapido de resultados
st.markdown("""
<div style='background-color: #2a2a2f; color: white; padding: 15px; border-radius: 10px; border: 2px solid #555; margin-top: 10px;'>
    <h4 style='margin-top: 0; color: #4CAF50;'>Analisis Rapido</h4>
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;'>
        <div style='background: rgba(255, 107, 107, 0.2); padding: 15px; border-radius: 5px; text-align: center;'>
            <div style='font-size: 32px; font-weight: bold; color: #FF6B6B;'>{0:.1f} C</div>
            <div style='font-size: 12px; color: #ccc;'>Aumento de Temperatura</div>
            <div style='font-size: 11px; margin-top: 5px;'>Reaccion {1}</div>
        </div>
        <div style='background: rgba(255, 217, 61, 0.2); padding: 15px; border-radius: 5px; text-align: center;'>
            <div style='font-size: 32px; font-weight: bold; color: #FFD93D;'>{2:.3f}</div>
            <div style='font-size: 12px; color: #ccc;'>Caida de Presion (bar)</div>
            <div style='font-size: 11px; margin-top: 5px;'>{3:.1f}% de P entrada</div>
        </div>
        <div style='background: rgba(107, 207, 127, 0.2); padding: 15px; border-radius: 5px; text-align: center;'>
            <div style='font-size: 32px; font-weight: bold; color: #6BCF7F;'>{4:.1f}%</div>
            <div style='font-size: 12px; color: #ccc;'>Conversion Alcanzada</div>
            <div style='font-size: 11px; margin-top: 5px;'>Eficiencia del reactor</div>
        </div>
    </div>
</div>
""".format(
    T_profile[-1] - T_inlet,
    'Exotermica' if T_profile[-1] > T_inlet else 'Endotermica',
    delta_P,
    (delta_P / P_inlet) * 100,
    conversion[-1]
), unsafe_allow_html=True)

# Separador visual
st.markdown("<br>", unsafe_allow_html=True)

# Información adicional sobre el empaquetamiento
with st.expander("Informacion del Reactor y Componentes", expanded=False):
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("### Lecho Empacado")
        st.markdown(f"""
        - **Partículas generadas:** {len(particles_simple)}
        - **Diámetro de partícula:** {d_particle*1000:.2f} mm
        - **Altura mínima:** {np.min(particles_simple[:, 2]):.3f} m
        - **Altura máxima:** {np.max(particles_simple[:, 2]):.3f} m
        - **Rango de presión:** {min(colors_pressure):.2f} - {max(colors_pressure):.2f} bar
        """)
    
    with col_info2:
        st.markdown("### 🔧 Instrumentación")
        st.markdown(f"""
        - **Manómetros:** 3 unidades (amarillo ●)
          - Superior: {L_reactor * 0.9:.2f} m
          - Medio: {L_reactor * 0.5:.2f} m
          - Inferior: {L_reactor * 0.1:.2f} m
        - **Termopozos:** 2 unidades (rojo ◆)
          - Superior: {L_reactor * 0.8:.2f} m
          - Inferior: {L_reactor * 0.3:.2f} m
        """)
    
    with col_info3:
        st.markdown("### Condiciones")
        st.markdown(f"""
        - **Entrada (azul ▲):**
          - T: {T_inlet:.1f}°C
          - P: {P_inlet:.1f} bar
        - **Salida (rojo ▲):**
          - T: {T_profile[-1]:.1f}°C
          - P: {P_profile[-1]:.1f} bar
        - **ΔT:** {T_profile[-1] - T_inlet:.1f}°C
        - **ΔP:** {delta_P:.3f} bar
        """)
    
    st.markdown("---")
    st.markdown("""
    **Interactividad:**
    - Pase el mouse sobre cualquier componente para ver informacion detallada
    - Use los botones para rotar automaticamente o resetear la vista
    - Haga zoom con la rueda del mouse
    - Arrastre para rotar manualmente
    - Los colores de las particulas indican presion (Viridis: violeta=baja, amarillo=alta)
    - Las isosuperficies muestran gradiente de concentracion en 3D
    """)

# Separador visual grande
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)

# Sección de análisis avanzado
st.subheader("Analisis Multiescala")
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Hidrodinámica", "Transferencia de Masa", "Cinética"])

with tab1:
    st.markdown("### Análisis Hidrodinámico")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Número de Reynolds
        Re_particle = rho_fluid * v_superficial * d_particle / 1.8e-5
        
        st.write("**Ecuación de Ergun:**")
        st.latex(r"\frac{\Delta P}{L} = \frac{150\mu v_s(1-\varepsilon)^2}{\varepsilon^3 d_p^2} + \frac{1.75\rho v_s^2(1-\varepsilon)}{\varepsilon^3 d_p}")
        
        st.write(f"- **Re partícula:** {Re_particle:.1f}")
        st.write(f"- **Régimen:** {'Laminar' if Re_particle < 10 else 'Transición' if Re_particle < 1000 else 'Turbulento'}")
        st.write(f"- **Caída de presión:** {delta_P:.4f} bar")
    
    with col_b:
        # Gráfico de contribuciones de Ergun
        term1 = 150 * 1.8e-5 * v_superficial * (1 - porosity)**2 / (porosity**3 * d_particle**2)
        term2 = 1.75 * rho_fluid * v_superficial**2 * (1 - porosity) / (porosity**3 * d_particle)
        
        fig_ergun = go.Figure(data=[
            go.Bar(name='Viscoso', x=['Contribución'], y=[term1/(term1+term2)*100]),
            go.Bar(name='Inercial', x=['Contribución'], y=[term2/(term1+term2)*100])
        ])
        fig_ergun.update_layout(
            title='Contribuciones a la Caída de Presión',
            yaxis_title='Porcentaje (%)',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig_ergun, use_container_width=True)

with tab2:
    st.markdown("### Transferencia de Masa")
    
    # Difusión efectiva
    D_eff = st.slider("Difusividad efectiva (m²/s)", 1e-9, 1e-6, 1e-7, format="%.2e")
    
    st.write("**Balance de masa en partícula esférica:**")
    st.latex(r"\frac{\partial C}{\partial t} = D_{eff}\left(\frac{\partial^2 C}{\partial r^2} + \frac{2}{r}\frac{\partial C}{\partial r}\right) - r_{rxn}")
    
    # Número de Thiele
    k_rxn = arrhenius_rate(T_K, k0, Ea)
    phi = (d_particle/2) * np.sqrt(k_rxn / D_eff)
    eta = (3/phi) * (1/np.tanh(phi) - 1/phi) if phi > 0.1 else 1.0
    
    col_c, col_d = st.columns(2)
    with col_c:
        st.metric("Módulo de Thiele (φ)", f"{phi:.3f}")
        st.metric("Factor de Efectividad (η)", f"{eta:.3f}")
    
    with col_d:
        # Gráfico de efectividad
        phi_range = np.logspace(-1, 2, 100)
        eta_range = np.array([(3/p) * (1/np.tanh(p) - 1/p) if p > 0.1 else 1.0 for p in phi_range])
        
        fig_thiele = go.Figure()
        fig_thiele.add_trace(go.Scatter(x=phi_range, y=eta_range, mode='lines',
                                        line=dict(color='green', width=3)))
        fig_thiele.add_trace(go.Scatter(x=[phi], y=[eta], mode='markers',
                                        marker=dict(size=15, color='red'),
                                        name='Punto actual'))
        fig_thiele.update_xaxes(type="log", title="Módulo de Thiele")
        fig_thiele.update_yaxes(title="Factor de Efectividad")
        fig_thiele.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_thiele, use_container_width=True)

with tab3:
    st.markdown("### Cinética Química")
    
    st.write("**Ecuación de Arrhenius:**")
    st.latex(r"k = k_0 \exp\left(-\frac{E_a}{RT}\right)")
    
    # Gráfico de Arrhenius
    T_range = np.linspace(273, 873, 100)
    k_range = arrhenius_rate(T_range, k0, Ea)
    
    fig_arr = go.Figure()
    fig_arr.add_trace(go.Scatter(x=T_range-273.15, y=k_range, mode='lines',
                                 line=dict(color='orange', width=3)))
    fig_arr.add_trace(go.Scatter(x=[T_inlet], y=[k_reaction], mode='markers',
                                 marker=dict(size=15, color='red'),
                                 name='Condición actual'))
    fig_arr.update_xaxes(title="Temperatura (°C)")
    fig_arr.update_yaxes(title="Constante de velocidad k (1/s)", type="log")
    fig_arr.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_arr, use_container_width=True)
    
    st.info(f"""
    **Parámetros cinéticos:**
    - Factor pre-exponencial: {k0:.2e} 1/s
    - Energía de activación: {Ea} kJ/mol
    - Constante a {T_inlet}°C: {k_reaction:.2e} 1/s
    """)

# Exportar datos
st.subheader("💾 Exportar Resultados")

# Crear DataFrame con resultados
df_results = pd.DataFrame({
    'Posición (m)': z_points,
    'Concentración (mol/m³)': C_profile,
    'Temperatura (°C)': T_profile,
    'Conversión (%)': conversion,
    'Presión (bar)': P_profile
})

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="📥 Descargar CSV",
        data=csv,
        file_name="resultados_reactor.csv",
        mime="text/csv"
    )

with col_exp2:
    st.write(f"**Resumen:** {len(z_points)} puntos de datos")

# Footer
st.markdown("---")
st.markdown("""
**Simulador de Reactor de Lecho Empacado 3D** | Basado en modelado multiescala CFD  
Incluye: Ecuación de Ergun, Cinética de Arrhenius, Balances de Masa y Energía, Análisis de Thiele
""")
