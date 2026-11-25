"""ExpensasAR. Aplicación local de consulta de liquidaciones procesadas."""
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
import re
import unicodedata

import altair as alt
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "02_Datos_Procesados"
INICIO_RECURRENTE = "2025-05"
FUENTE_IPC = "https://www.indec.gob.ar/ftp/cuadros/economia/serie_ipc_divisiones.csv"
MESES = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
TIPOS_GASTOS_UF = (
    ("reparacion", "Reparaciones", "Reparaciones / trabajos"),
    ("amenity", "Amenities", "Amenities / reservas"),
    ("gasto_particular", "Otros", "Otros cargos particulares"),
)

AZUL = "#233D60"
VERDE = "#1C786D"
OCRE = "#A55C31"
GRIS = "#65768C"
ROJO = "#A14F56"

# Equivalencias y reglas del notebook de la Etapa 2.
ALIAS_PROVEEDOR = {
    "MUNICIP SAN ISIDRO": "MUNICIPALIDAD DE SAN ISIDRO",
    "SULIMP": "SULIMP SA",
    "ARBA": "AGENCIA DE RECAUDACION DE LA PROVINCIA DE BUENOS AIRES",
}
RUBROS_EXCLUIDOS = {
    "EGRESOS EXTRAORDINARIOS Y NO RECURRENTES",
    "GASTOS EXTRAORDINARIOS",
    "GASTOS ORDINARIOS NO RECURRENTES",
}
CONCEPTO_TRABAJO_EXTRAORDINARIO = re.compile(
    r"\b(?:repar\w*|filtraci\w*|impermeabili\w*|reacondicion\w*|reemplaz\w*|"
    r"sellad\w*|destap\w*|humedad\w*|trabajo\w*|calder\w*|mamposter\w*|"
    r"masill\w*|endui\w*|pintur\w*|pintad\w*|pintar|lijad\w*|sellar|desamurar|"
    r"limpiar hongos|amurad\w*|colocacion|instalacion|excavacion|enderezad\w*|"
    r"recolocacion|modificacion|recambio|cambio de|desarmad\w*|empalm\w*|regulacion|"
    r"ajuste herreria|saldo presupuesto UF|se coloco el porton en guia y se regulo)\b",
    re.IGNORECASE,
)
INSUMOS_O_MANTENIMIENTO_GENERAL = re.compile(
    r"^(?:PROVISION DE INSUMOS DE PINTURA|SERVICIO DE MANTENIMIENTO DE TRES CALDERAS|"
    r"LIMPIEZA Y MANTENIMIENTO DE 3 CALDERAS)"
)


def numero(valor, decimales=1):
    return f"{valor:,.{decimales}f}".translate(str.maketrans(",.", ".,"))


def dinero(valor, abreviado=True):
    if pd.isna(valor):
        return "Sin dato"
    if abreviado:
        return "$" + numero(valor / 1e6) + " M"
    return "$" + numero(valor, 2)


def millones(valor):
    return f"${numero(valor / 1e6)} millones" if pd.notna(valor) else "sin dato"


def porcentaje(valor, con_signo=False):
    if pd.isna(valor):
        return "Sin dato"
    signo = "+" if con_signo and valor > 0 else ""
    return f"{signo}{numero(valor)} %"


def nombre_mes(mes):
    return f"{MESES[int(mes[5:7]) - 1]} de {mes[:4]}"


def clave(texto):
    letras = unicodedata.normalize("NFD", texto.upper())
    return "".join(letra for letra in letras if not unicodedata.combining(letra))


def proveedor_id(nombre):
    if pd.isna(nombre):
        return None
    etiqueta = " ".join(re.sub(r"[^A-Z0-9 ]", " ", clave(nombre)).split())
    return "PROV_" + ALIAS_PROVEEDOR.get(etiqueta, etiqueta).replace(" ", "_")


@st.cache_data
def leer_csv(ruta, version):
    # La fecha de modificación invalida la caché cuando cambia un CSV.
    tipos = dict.fromkeys(("mes", "pagina_origen", "pagina_cargo_vinculado", "pagina_plan_de_pagos", "Periodo"), str)
    return pd.read_csv(ruta, dtype=tipos,
                       sep=";" if ruta.name.startswith("ipc_") else ",",
                       decimal="," if ruta.name.startswith("ipc_") else ".")


def cargar(nombre):
    ruta = DATOS / f"{nombre}.csv"
    return leer_csv(ruta, ruta.stat().st_mtime_ns)


def calcular_rubros(detalle, meses):
    importes = detalle.pivot_table(
        index="mes", columns="rubro", values="importe", aggfunc="sum", fill_value=0,
    ).reindex(meses)
    total = importes.sum(axis=1, min_count=1)
    cambio = importes.diff()
    contribucion = cambio.div(total.shift().where(total.shift().ne(0)), axis=0) * 100
    return importes, total, cambio, contribucion


def calcular_proveedores(detalle, maestro, total):
    con_nombre = detalle.assign(proveedor_id=detalle["proveedor"].map(proveedor_id))
    mensual = con_nombre.groupby(["mes", "proveedor_id"], as_index=False)["importe"].sum()
    mensual["participacion_pct"] = mensual["importe"] / mensual["mes"].map(total) * 100
    mensual = mensual.join(maestro.set_index("proveedor_id")[["nombre", "categoria_predominante"]], on="proveedor_id")
    return mensual.sort_values(["mes", "importe", "nombre"], ascending=[True, False, True])


def concentracion_trabajos_extraordinarios(detalle):
    # Revisión por concepto. Incluye intervenciones y desembolsos asociados a obras
    # explícitas. Excluye compras sueltas, trámites y mantenimiento general.
    rubro = detalle["categoria_original"].fillna("").map(clave)
    conceptos = detalle["concepto"].fillna("").map(clave)
    movimientos = detalle.loc[
        rubro.isin(RUBROS_EXCLUIDOS)
        & detalle["base_registro"].eq("pago")
        & conceptos.str.contains(CONCEPTO_TRABAJO_EXTRAORDINARIO)
        & ~conceptos.str.contains(INSUMOS_O_MANTENIMIENTO_GENERAL),
        ["proveedor", "mes", "importe"],
    ].copy()
    movimientos["proveedor_id"] = movimientos["proveedor"].map(proveedor_id).fillna("sin_proveedor")
    movimientos["proveedor"] = movimientos["proveedor"].fillna("Sin proveedor identificado")
    ranking = movimientos.groupby("proveedor_id").agg(
        Proveedor=("proveedor", lambda nombres: nombres.value_counts().idxmax()),
        Movimientos=("importe", "size"),
        Importe=("importe", "sum"),
        meses_activos=("mes", "nunique"),
        primer_mes=("mes", "min"),
        ultimo_mes=("mes", "max"),
    )
    ranking["promedio_activo"] = ranking["Importe"] / ranking["meses_activos"]
    total = ranking["Importe"].sum()
    ranking["Participación"] = ranking["Importe"] / (total or float("nan")) * 100
    return ranking.sort_values(["Importe", "Proveedor"], ascending=[False, True]).reset_index(drop=True)


def comparar_ipc(resumen, detalle, ipc):
    gba = ipc.loc[ipc["Region"].eq("GBA")].copy()
    gba["mes"] = pd.to_datetime(gba["Periodo"], format="%Y%m").dt.strftime("%Y-%m")
    nivel = gba.set_index("mes")["Indice_IPC"].sort_index()
    tabla = resumen[["gastos_informados"]].copy()
    tabla["indice_ipc"] = nivel
    tabla["ipc_mensual_pct"] = nivel.pct_change(fill_method=None) * 100
    excluido = detalle["categoria_original"].fillna("").map(clave).isin(RUBROS_EXCLUIDOS)
    excepcionales = detalle.loc[excluido].groupby("mes")["importe"].sum()
    tabla["gasto_recurrente"] = (
        tabla["gastos_informados"] - excepcionales.reindex(tabla.index, fill_value=0)
    ).where(tabla.index >= INICIO_RECURRENTE)
    tabla["variacion_total_pct"] = tabla["gastos_informados"].pct_change(fill_method=None) * 100
    tabla["variacion_recurrente_pct"] = tabla["gasto_recurrente"].pct_change(fill_method=None) * 100
    return tabla


def calcular_alertas(detalle, resumen, ipc):
    # Solo meses comparables, con seis observaciones anteriores y desvío positivo.
    nominal = detalle.groupby(["mes", "rubro"])["importe"].sum().unstack().reindex(resumen.index)
    real = nominal.div(ipc / ipc.iloc[0], axis=0)
    media = real.shift().rolling(6, min_periods=6).mean()
    desvio = real.shift().rolling(6, min_periods=6).std(ddof=1)
    z = (real - media).div(desvio.where(desvio.gt(0)))
    misma_admin = resumen["administracion"].eq(resumen["administracion"].shift(6))
    z.loc[~misma_admin, :] = float("nan")
    return pd.concat({"gasto_real": real.stack(), "z_score": z.stack()}, axis=1).reset_index()


def explicar_caja(fila):
    partes = []
    for campo, frase, etiqueta in [
        ("saldo_inicial", "La caja abrió con {}", "Saldo inicial"),
        ("expensas_cobradas", "Se registraron {} de expensas cobradas", "Expensas cobradas"),
        ("pagos_realizados", "Se realizaron pagos por {}", "Pagos realizados"),
        ("saldo_final", "La caja cerró con {}", "Saldo final"),
    ]:
        valor = fila[campo]
        partes.append(frase.format(millones(valor)) if pd.notna(valor) else f"{etiqueta} sin dato")
    return ". ".join(partes) + "."


# Relato de la Etapa 2, adaptado a la selección de mes.
def resumir_contexto(mes):
    fila = contexto_mensual.loc[mes]
    nota = fila["contexto_documentado"]
    if pd.isna(nota):
        return "La revisión de las notas no encontró contexto económico específico para este mes."
    paginas = ", ".join(fila["pagina_origen"].split("|"))
    prefijo = "pp." if "," in paginas else "p."
    return f"{nota} Fuente: {fila['archivo_origen']}, {prefijo} {paginas}."


def frase_variacion(valor):
    if pd.isna(valor):
        return "sin comparación mensual disponible"
    if abs(valor) < 0.0001:
        return "sin variación"
    direccion = "más" if valor > 0 else "menos"
    return f"un {numero(abs(valor))} % {direccion} que el mes anterior"

def resumir_cambio_rubros(mes):
    cambios = cambio_rubro.loc[mes].dropna()
    if cambios.empty:
        return ""
    partes = []
    if cambios.max() > 0:
        partes.append(f"El principal aumento en pesos se observó en {cambios.idxmax().capitalize()} ({millones(cambios.max())}).")
    if cambios.min() < 0:
        partes.append(f"La mayor baja se observó en {cambios.idxmin().capitalize()} ({millones(abs(cambios.min()))}).")
    return " ".join(partes)

def resumen_breve(mes, fila, variacion):
    gasto = (
        f"En {nombre_mes(mes)}, el consorcio informó gastos por "
        f"{millones(fila['gastos_informados'])}, {frase_variacion(variacion)}."
    )
    cambios = cambio_rubro.loc[mes].dropna()
    if cambios.empty:
        rubro = f"El rubro de mayor importe fue {rubros.loc[mes].idxmax().capitalize()}."
    else:
        principal = cambios.abs().idxmax()
        cambio = cambios.loc[principal]
        direccion = "subió" if cambio > 0 else "bajó"
        rubro = (
            "Los importes por rubro se mantuvieron."
            if cambio == 0
            else f"El mayor cambio en pesos se observó en {principal.capitalize()}, que {direccion} {millones(abs(cambio))}."
        )
    saldo = fila["saldo_final"]
    cierre = f"La caja cerró con {millones(saldo)}." if pd.notna(saldo) else "El saldo final de caja figura sin dato."
    return " ".join([gasto, rubro, cierre])


def resumir_operaciones(mes):
    top = top_5_mes[top_5_mes["mes"] == mes].head(3)
    if len(top):
        nombres = ", ".join(top["nombre"].str.title())
        proveedores_texto = f"Los tres mayores proveedores ({nombres}) reunieron {numero(top['participacion_pct'].sum())} % del gasto. "
    else:
        proveedores_texto = ""
    reparaciones_mes = reparaciones[reparaciones["mes"] == mes]
    if len(reparaciones_mes):
        principal = reparaciones_mes.groupby("unidad_funcional")["importe"].sum().idxmax()
        reparaciones_texto = (
            f"Se registraron {len(reparaciones_mes)} movimientos de reparación vinculados a UF "
            f"por {millones(reparaciones_mes['importe'].sum())}. "
            f"El mayor importe registrado se vinculó a {principal.replace('_', ' ')}. "
        )
        if reparaciones_mes["alcance_importe"].notna().any():
            reparaciones_texto += "Algunos pagos también incluyen trabajos en otra unidad o en sectores comunes. "
    else:
        reparaciones_texto = "La tabla de unidades carece de reparaciones vinculadas a una UF numérica en este mes. "
    casos = revision["mes"].eq(mes).sum()
    pendientes_texto = {
        0: "La tabla de revisión no registra casos para este mes.",
        1: "Hay un movimiento que requiere explicación adicional.",
    }.get(casos, f"Hay {casos} movimientos que requieren explicación adicional.")
    return proveedores_texto + reparaciones_texto + pendientes_texto

def resumir_ipc(mes):
    fila = comparacion.loc[mes]
    if pd.isna(fila["ipc_mensual_pct"]):
        return "La comparación mensual con el IPC GBA queda pendiente."
    tasa = fila["variacion_total_pct"]
    ipc = fila["ipc_mensual_pct"]
    partes = [f"El IPC GBA varió {numero(ipc)} % en el mes."]
    if pd.notna(tasa):
        partes.append(f"La brecha del gasto informado frente al IPC fue de {numero(tasa - ipc)} puntos porcentuales.")
    if pd.notna(fila["variacion_recurrente_pct"]):
        partes.append(f"El gasto recurrente aproximado varió {numero(fila['variacion_recurrente_pct'])} %.")
    partes.append("La comparación pone en contexto los importes y puede reflejar precios, cantidades o calendario de pagos.")
    return " ".join(partes)

def generar_resumen_mes(mes):
    mes = str(pd.Period(mes, freq="M"))
    if mes not in resumen.index:
        raise ValueError("Elegí un mes disponible en la serie.")
    fila = resumen.loc[mes]
    variacion = comparacion.loc[mes, "variacion_total_pct"]
    principales = rubros.loc[mes].nlargest(3)
    nombres = ", ".join(principales[principales.gt(0)].index.str.capitalize())
    datos_texto = (
        f"En {nombre_mes(mes)}, el gasto informado fue de {millones(fila['gastos_informados'])}, "
        f"{frase_variacion(variacion)}. Los rubros de mayor importe fueron {nombres}. "
        f"{resumir_cambio_rubros(mes)} {explicar_caja(fila)} {resumir_operaciones(mes)}"
    )
    lectura = resumir_ipc(mes)
    if mes == "2025-05":
        lectura += " La transición cambió el alcance de los saldos y la clasificación de algunos rubros."
    elif pd.notna(contexto_mensual.loc[mes, "contexto_documentado"]):
        lectura += " La nota aporta contexto sobre la liquidación. Su aporte al cambio del gasto no está cuantificado."
    return (
        "Datos observados:\n" + datos_texto + "\n\n"
        "Contexto informado:\n" + resumir_contexto(mes) + "\n\n"
        "Lectura:\n" + lectura
    )


def mostrar_texto(texto):
    st.markdown(texto.replace("$", r"\$"))


def tabla_monetaria(tabla, columnas):
    tabla = tabla.copy()
    for columna in columnas:
        tabla[columna] = tabla[columna].map(lambda valor: dinero(valor, abreviado=False))
    st.dataframe(tabla, hide_index=True, width="stretch")


def tabla_gastos_uf(movimientos, *, con_cargo_vinculado=False):
    campos = {
        "unidad_funcional": "UF", "portal": "Portal", "piso": "Piso",
        "departamento": "Departamento", "concepto": "Concepto", "importe": "Importe",
        "base_registro": "Registro", "proveedor": "Proveedor",
        "archivo_origen": "Archivo de origen", "pagina_origen": "Página",
        "seccion_origen": "Sección de origen", "trabajo_id": "Trabajo vinculado",
    }
    if movimientos["alcance_importe"].notna().any():
        campos.update(uf_relacionada="UF relacionada", alcance_importe="Alcance del pago")
    if con_cargo_vinculado:
        campos.update(archivo_cargo_vinculado="Archivo del cargo", pagina_cargo_vinculado="Página del cargo")
    tabla = movimientos[list(campos)].rename(columns=campos)
    tabla["UF"] = tabla["UF"].str.replace("_", " ", regex=False)
    tabla["Registro"] = tabla["Registro"].replace({
        "cargo_uf": "Cargo liquidado a la UF", "pago": "Pago del consorcio",
        "gasto_informado": "Gasto informado", "pendiente": "Pendiente de pago",
    })
    tabla_monetaria(tabla, ["Importe"])


def mostrar_pagos_vinculados_uf(pagos):
    if not pagos.empty:
        with st.expander("Pagos vinculados a cargos de otro mes"):
            st.caption("Estos pagos respaldan cargos ya registrados en otra liquidación. "
                       "Los conteos e importes de esta sección toman el cargo original una sola vez.")
            tabla_gastos_uf(pagos, con_cargo_vinculado=True)


def mostrar_gastos_y_prorrateo(mes, fila):
    publicados = controles_prorrateo.loc[
        controles_prorrateo["mes"].eq(mes) & controles_prorrateo["estado"].eq("OK")
    ].set_index("codigo_concepto")["total_pdf"]
    prorrateado = publicados.get("PERIODO")
    gasto = fila["gastos_informados"]
    if pd.isna(prorrateado) or pd.isna(gasto):
        return
    if round(fila["expensas_emitidas"], 2) != prorrateado or round(total_por_mes.loc[mes], 2) != gasto:
        return

    diferencia = round(abs(gasto - prorrateado), 2)
    detalle = [
        ("Gastos del mes", gasto),
        ("Importe fijado a prorratear", prorrateado),
        ("Diferencia absoluta", diferencia),
    ]
    with st.container(border=True):
        st.subheader("Gastos del mes y prorrateo")
        mes_corto = nombre_mes(mes).split(" de ", 1)[0]
        mostrar_texto(f"En {mes_corto}, el consorcio registró **${numero(gasto / 1e6, 2)} M de gastos** "
                      f"y fijó **${numero(prorrateado / 1e6, 2)} M a prorratear** entre las UF.")
        mostrar_texto(f"**Diferencia absoluta · {dinero(diferencia, abreviado=False)}.**")
        st.caption("Los gastos del mes y el importe liquidado no siguen necesariamente una relación uno a uno.")

        if mes == "2026-10":
            composicion = publicados.reindex(["A", "C", "F"])
            if composicion.notna().all() and round(composicion.sum(), 2) == prorrateado:
                detalle += [
                    ("A — Expensa ordinaria", composicion["A"]),
                    ("C — Impuesto municipal", composicion["C"]),
                    ("F — Prorrateo fijo extraordinario/no recurrente", composicion["F"]),
                    ("Total del prorrateo A + C + F", prorrateado),
                ]
                extraordinarios = gastos.loc[
                    gastos["mes"].eq(mes) & gastos["base_registro"].eq("pago")
                    & gastos["categoria_original"].eq("EGRESOS EXTRAORDINARIOS Y NO RECURRENTES"), "importe",
                ].sum(min_count=1)
                if pd.notna(extraordinarios):
                    columna_f = composicion["F"]
                    brecha_f = round(extraordinarios - columna_f, 2)
                    mostrar_texto(f"Durante octubre se pagaron **${numero(extraordinarios / 1e6, 2)} M** en egresos "
                                  f"extraordinarios/no recurrentes, mientras que la columna F prorrateó **${numero(columna_f / 1e6, 2)} M**.")
                    if brecha_f > 0:
                        mostrar_texto(f"La diferencia de **${numero(brecha_f / 1e6, 2)} M** representa extraordinarios "
                                      "pagados durante el mes que no fueron trasladados íntegramente mediante F en esa misma liquidación.")
                        st.write("Este descalce contribuyó a presionar la liquidez del consorcio durante octubre. "
                                 "No explica por sí solo la caída de caja y la liquidación no permite identificar qué fuente específica financió cada gasto extraordinario.")
                    detalle += [
                        ("Egresos extraordinarios y no recurrentes pagados", extraordinarios),
                        ("Diferencia entre extraordinarios pagados y F", brecha_f),
                    ]
                st.caption("La liquidación concilia completamente los movimientos de caja, pero no presenta una conciliación específica entre gastos pagados, egresos devengados y montos finalmente prorrateados.")

        with st.expander("Ver importes exactos del gasto y el prorrateo"):
            tabla_monetaria(pd.DataFrame(detalle, columns=["Concepto", "Importe"]), ["Importe"])


def desglosar_ordinaria(mapeo_mes, codigo, importe_uf, total_concepto):
    centavo = Decimal("0.01")
    importe_uf = Decimal(str(importe_uf)).quantize(centavo)
    total_concepto = Decimal(str(total_concepto)).quantize(centavo)
    por_rubro = mapeo_mes.loc[mapeo_mes["codigo_concepto"].eq(codigo)].groupby("rubro")["importe_consorcio"].sum()
    bases = {rubro: Decimal(str(importe)).quantize(centavo) for rubro, importe in por_rubro.items() if importe}
    base_identificada = sum(bases.values())
    if total_concepto <= 0 or importe_uf < 0 or not bases or base_identificada > total_concepto:
        return None
    if base_identificada < total_concepto:
        bases["Otros gastos ordinarios"] = total_concepto - base_identificada

    coeficiente = importe_uf / total_concepto
    cuotas = [base * coeficiente / centavo for base in bases.values()]
    montos = [int(cuota.to_integral_value(rounding=ROUND_FLOOR)) for cuota in cuotas]
    # Distribuye los centavos de redondeo a los rubros con mayor parte
    # fraccionaria para preservar el importe exacto publicado de la UF.
    faltan = int(importe_uf / centavo) - sum(montos)
    orden = sorted(range(len(cuotas)), key=lambda i: cuotas[i] - montos[i], reverse=True)
    for indice in orden[:faltan]:
        montos[indice] += 1
    tabla = pd.DataFrame({
        "Rubro": [rubro.capitalize() if rubro.isupper() else rubro for rubro in bases],
        "Importe en la UF": [float(centavo * monto) for monto in montos],
    })
    return tabla.sort_values("Importe en la UF", ascending=False), coeficiente


def mostrar_expensa_uf(mes, uf):
    periodos = liquidaciones_uf.loc[liquidaciones_uf["uf"].eq(uf)].set_index("mes")
    if mes not in periodos.index:
        st.info("La liquidación seleccionada no tiene un total del período verificable para esta UF.")
        return
    periodo = periodos.loc[mes]
    anterior = str(pd.Period(mes, freq="M") - 1)
    variacion = None
    if anterior in periodos.index:
        previo = periodos.loc[anterior]
        formato_anterior = previo["concepto_total_publicado"] == "TOTAL"
        formato_actual = periodo["concepto_total_publicado"] == "TOTAL"
        if previo["expensa_prorrateada"] > 0 and formato_anterior == formato_actual:
            variacion = (periodo["expensa_prorrateada"] / previo["expensa_prorrateada"] - 1) * 100
    conceptos = prorrateos.loc[prorrateos["mes"].eq(mes) & prorrateos["uf"].eq(uf)]
    ordinario = conceptos.loc[conceptos["codigo_concepto"].isin(["A", "ORD"])]
    redondeo = periodo["redondeo"]
    if pd.isna(redondeo):
        ajuste = "Sin ajuste publicado"
    else:
        signo = "+" if redondeo > 0 else ("−" if redondeo < 0 else "")
        ajuste = signo + dinero(abs(redondeo), abreviado=False)
    with st.container(border=True):
        st.subheader("Tu liquidación")
        mostrar_texto(f"Expensa prorrateada · {dinero(periodo['expensa_prorrateada'], abreviado=False)}")
        mostrar_texto(f"Gastos particulares · {dinero(periodo['gastos_particulares'], abreviado=False)}")
        mostrar_texto(f"Redondeo · {ajuste}")
        st.metric("Total liquidado del período", dinero(periodo["total_reconstruido"], abreviado=False))
        st.caption(f"{periodo['archivo_origen']}, pág. {periodo['pagina_origen']} · "
                   f"Columna de referencia: {periodo['concepto_total_publicado']}.")
        diferencia = periodo["diferencia_publicado_menos_reconstruido"]
        if diferencia:
            mostrar_texto(f"**Importe a pagar publicado · {dinero(periodo['total_publicado'], abreviado=False)}**")
            with st.expander("Ver diferencia con el importe a pagar publicado"):
                mostrar_texto(f"Diferencia (publicado menos total del período) · {dinero(diferencia, abreviado=False)}")
                for campo, nombre in (
                    ("deuda_publicada", "Deuda o saldo a favor"),
                    ("interes_publicado", "Intereses"),
                    ("plan_de_pagos", "Plan de pagos"),
                ):
                    if pd.notna(periodo[campo]) and periodo[campo]:
                        mostrar_texto(f"{nombre} · {dinero(periodo[campo], abreviado=False)}")
                if pd.notna(periodo["pagina_plan_de_pagos"]):
                    st.caption(f"El plan figura en el detalle de {periodo['archivo_origen']}, "
                               f"pág. {periodo['pagina_plan_de_pagos']}.")
                if periodo["diferencia_tras_otros_conceptos"]:
                    mostrar_texto("Diferencia que queda por conciliar · "
                                  + dinero(periodo["diferencia_tras_otros_conceptos"], abreviado=False))
        else:
            st.caption("La suma coincide al centavo con el importe publicado.")

    columnas = st.columns(2)
    columnas[0].metric("Variación mensual del prorrateo", porcentaje(variacion, con_signo=True) if variacion is not None else "Sin comparación")
    columnas[1].metric(
        "Incidencia ordinaria publicada",
        f"{numero(ordinario.iloc[0]['porcentaje_incidencia'], 2)} %" if len(ordinario) else "Sin dato",
    )

    st.subheader("Desglose del prorrateo")
    mapeo_mes = mapeo_prorrateo[mapeo_prorrateo["mes"].eq(mes)]
    if not ordinario.empty:
        fila_ordinaria = ordinario.iloc[0]
        codigo = fila_ordinaria["codigo_concepto"]
        importe = fila_ordinaria["importe_asignado"]
        mostrar_texto(f"**Expensa ordinaria · {dinero(importe, abreviado=False)}**")
        st.caption(f"{codigo} · {fila_ordinaria['concepto_publicado']} · "
                   f"{fila_ordinaria['archivo_origen']}, pág. {fila_ordinaria['pagina_origen']}.")
        total = controles_prorrateo.loc[
            controles_prorrateo["mes"].eq(mes) & controles_prorrateo["codigo_concepto"].eq(codigo)
            & controles_prorrateo["estado"].eq("OK"), "total_pdf",
        ]
        desglose = desglosar_ordinaria(mapeo_mes, codigo, importe, total.iloc[0]) if len(total) else None
        if desglose is None:
            st.info("La liquidación respalda el importe ordinario. Los gastos registrados del mes no permiten desglosarlo de forma fiable.")
        else:
            tabla, coeficiente = desglose
            st.caption(f"Coeficiente efectivo de {numero(coeficiente * 100, 6)} % para este desglose. "
                       "Surge de dividir la ordinaria de la UF por el total ordinario publicado del consorcio.")
            st.write("Los rubros se estiman a partir del detalle de gastos registrados. "
                     "«Otros gastos ordinarios» es la parte del total ordinario publicado que carece de una apertura verificable por rubro.")
            tabla_monetaria(tabla, ["Importe en la UF"])
            mostrar_texto(f"**Suma de la expensa ordinaria · {dinero(importe, abreviado=False)}**")

    otros = conceptos.loc[~conceptos["codigo_concepto"].isin(["A", "ORD"])]
    if not otros.empty:
        st.markdown("**Otros conceptos prorrateados**")
        conceptos_pdf = otros.rename(columns={
            "codigo_concepto": "Columna", "concepto_publicado": "Concepto publicado",
            "importe_asignado": "Importe liquidado", "archivo_origen": "Archivo de origen", "pagina_origen": "Página",
        })
        tabla_monetaria(conceptos_pdf[["Columna", "Importe liquidado", "Concepto publicado", "Archivo de origen", "Página"]],
                        ["Importe liquidado"])

    sin_asignar = mapeo_mes.loc[~mapeo_mes["codigo_concepto"].isin(conceptos["codigo_concepto"])]
    if not sin_asignar.empty:
        with st.expander("Ver gastos registrados sin vínculo de prorrateo"):
            tabla = sin_asignar.rename(columns={
                "rubro": "Rubro", "criterio": "Motivo", "importe_consorcio": "Gasto del consorcio",
            })
            tabla.loc[tabla["codigo_concepto"].notna(), "Motivo"] = "El concepto carece de un coeficiente validado."
            tabla_monetaria(tabla[["Rubro", "Gasto del consorcio", "Motivo"]], ["Gasto del consorcio"])


def barras(tabla, campo, etiqueta, divergente=False):
    colores = alt.condition(alt.datum[campo] > 0, alt.value(ROJO), alt.value(VERDE)) if divergente else alt.value(AZUL)
    chart = alt.Chart(tabla).mark_bar(cornerRadiusEnd=3, size=17).encode(
        x=alt.X(f"{campo}:Q", title=etiqueta),
        y=alt.Y("rubro:N", title=None, sort=tabla["rubro"].tolist(), axis=alt.Axis(labelLimit=300)),
        color=colores,
        tooltip=["rubro:N", alt.Tooltip(f"{campo}:Q", title=etiqueta, format=".2f")],
    ).properties(height=max(110, 28 * len(tabla)))
    st.altair_chart(chart, width="stretch")


def variacion_destacada(valor, criterio="gasto"):
    texto = porcentaje(valor, con_signo=True)
    if pd.isna(valor) or valor == 0:
        return texto
    if criterio == "ipc":
        color = "orange"
    else:
        color = "red" if (valor > 0) == (criterio == "gasto") else "green"
    return f":{color}[{texto}]"


def variaciones_interanuales(rubros, indice_ipc):
    elegidos = ["PERSONAL", "Abonos y servicios", "Servicios públicos", "Impuestos", "Seguridad"]
    # El pivote mensual rellena ausencias con cero. Para comparar el mismo
    # rubro entre dos años exigimos un importe positivo en ambos meses.
    serie = rubros.reindex(columns=elegidos).where(lambda valores: valores.gt(0))
    serie["IPC GBA"] = indice_ipc.where(indice_ipc.gt(0))
    return (serie.pct_change(periods=12, fill_method=None) * 100).iloc[12:]


def grafico_interanual(variaciones, rubro):
    fechas = variaciones.index.tolist()
    nombre_rubro = rubro.capitalize()
    datos = variaciones[[rubro, "IPC GBA"]].rename(columns={rubro: nombre_rubro})
    datos = datos.rename_axis("mes").reset_index()
    puntos_mes = datos.melt("mes", var_name="serie", value_name="variacion")
    datos["mes_legible"] = datos["mes"].map(lambda m: nombre_mes(m).capitalize())
    datos["rubro_pct"] = datos[nombre_rubro].map(lambda v: porcentaje(v, con_signo=True))
    datos["ipc_pct"] = datos["IPC GBA"].map(lambda v: porcentaje(v, con_signo=True))
    datos["brecha"] = (datos[nombre_rubro] - datos["IPC GBA"]).map(
        lambda v: porcentaje(v, con_signo=True).replace(" %", " p.p.")
    )
    eje_mes = alt.X("mes:N", title="Mes de la comparación", sort=fechas, scale=alt.Scale(type="band"),
                    axis=alt.Axis(labelAngle=-35, labelFontSize=10))
    base = alt.Chart(puntos_mes).encode(
        x=eje_mes,
        y=alt.Y("variacion:Q", title="Cambio interanual (%)", scale=alt.Scale(type="linear", zero=True)),
        color=alt.Color("serie:N", scale=alt.Scale(domain=[nombre_rubro, "IPC GBA"], range=[AZUL, OCRE]),
                        legend=alt.Legend(title=None, orient="bottom")),
    )
    referencia = base.transform_filter(alt.datum.serie == "IPC GBA").mark_line(strokeWidth=2)
    puntos = base.mark_point(filled=True, size=85, opacity=1)
    consulta = alt.Chart(datos).mark_rect(fillOpacity=0).encode(
        x=eje_mes,
        tooltip=[
            alt.Tooltip("mes_legible:N", title="Mes"),
            alt.Tooltip("rubro_pct:N", title=nombre_rubro),
            alt.Tooltip("ipc_pct:N", title="IPC GBA"),
            alt.Tooltip("brecha:N", title="Diferencia (rubro − IPC)"),
        ],
    )
    st.altair_chart(
        (referencia + puntos + consulta).properties(height=270), width="stretch",
        alt=f"Evolución interanual del gasto en {nombre_rubro} y del IPC GBA",
    )


def rubros_para_ipc(rubros, mes):
    anterior = rubros.shift().loc[mes]
    actual = rubros.loc[mes]
    tabla = pd.DataFrame({"actual": actual, "anterior": anterior})
    tabla = tabla.loc[actual.ne(0) | anterior.fillna(0).ne(0)].copy()
    tabla["cambio"] = tabla["actual"] - tabla["anterior"]
    tabla["variacion"] = tabla["cambio"].div(tabla["anterior"].where(tabla["anterior"].ne(0))) * 100
    tabla["peso"] = tabla[["actual", "anterior"]].abs().max(axis=1)
    tabla["rubro"] = tabla.index.str.capitalize()
    tabla = tabla.rename_axis(None)
    return tabla.sort_values("peso", ascending=False)


def grafico_rubros_ipc(tabla, ipc_mensual):
    datos = tabla.sort_values("variacion", ascending=False).copy()
    if datos.empty:
        st.markdown("**Sin base comparable para los rubros principales.**")
        if pd.isna(ipc_mensual):
            return
    if pd.notna(ipc_mensual):
        referencia = {"rubro": "IPC GBA", "variacion": ipc_mensual}
        datos = pd.DataFrame([referencia]) if datos.empty else pd.concat(
            [datos, pd.DataFrame([referencia])], ignore_index=True
        )
    datos["etiqueta"] = datos["variacion"].map(lambda valor: porcentaje(valor, con_signo=True))
    datos["color"] = datos["variacion"].map(lambda v: ROJO if v > 0 else (VERDE if v < 0 else GRIS))
    datos.loc[datos["rubro"].eq("IPC GBA"), "color"] = OCRE
    datos["rubro_grafico"] = datos["rubro"].replace({
        "Egresos extraordinarios y no recurrentes": "Extraordinarios",
        "Mantenimiento de partes comunes": "Mant. partes comunes",
        "Gastos ordinarios no recurrentes": "Ordinarios no recurrentes",
    })
    magnitudes = datos["variacion"].abs().nlargest(2)
    valor_extremo = len(magnitudes) == 2 and magnitudes.iloc[0] >= 100 and magnitudes.iloc[0] > 5 * magnitudes.iloc[1]
    datos["altura"] = datos["variacion"]
    if valor_extremo:
        limite = max(50, 1.25 * magnitudes.iloc[1])
        datos["altura"] = datos["variacion"].clip(-limite, limite)
    datos["fuera_de_escala"] = datos["altura"].ne(datos["variacion"])
    datos["etiqueta_grafico"] = datos["etiqueta"]
    datos.loc[datos["fuera_de_escala"], "etiqueta_grafico"] += "\nFuera de escala"
    base = alt.Chart(datos).encode(
        x=alt.X("rubro_grafico:N", title=None, sort=datos["rubro_grafico"].tolist(),
                scale=alt.Scale(paddingInner=0.32, paddingOuter=0.25),
                axis=alt.Axis(labelAngle=-42, labelLimit=114, labelOverlap=False, labelFontSize=10)),
        y=alt.Y("altura:Q", title="Variación mensual (%)",
                scale=alt.Scale(type="linear", zero=True, padding=32 if valor_extremo else 20),
                axis=alt.Axis(labelExpr=r"replace(replace(replace(datum.label, /,/g, '|'), /\./g, ','), /\|/g, '.') + ' %'")),
    )
    barras = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color("color:N", scale=None),
        tooltip=[alt.Tooltip("rubro:N", title="Rubro"), alt.Tooltip("etiqueta:N", title="Variación")],
    )
    etiquetas = (
        base.transform_filter(alt.datum.variacion >= 0)
        .mark_text(baseline="bottom", dy=alt.ExprRef(expr="datum.fuera_de_escala ? -17 : -5"),
                   fontSize=10, color=AZUL, lineBreak="\n")
        .encode(text="etiqueta_grafico:N")
        + base.transform_filter(alt.datum.variacion < 0)
        .mark_text(baseline="top", dy=5, fontSize=10, color=AZUL, lineBreak="\n")
        .encode(text="etiqueta_grafico:N")
    )
    cero = alt.Chart(pd.DataFrame({"cero": [0]})).mark_rule(color=GRIS, opacity=0.5).encode(y="cero:Q")
    st.altair_chart(
        (barras + cero + etiquetas).properties(height=300 if valor_extremo else 280),
        alt="Variación mensual del gasto registrado por rubro y del IPC GBA en una escala lineal",
        width="stretch",
    )
    if pd.notna(ipc_mensual):
        st.caption("La columna ámbar muestra el IPC GBA.")
    if valor_extremo:
        st.caption("Fuera de escala indica una columna recortada visualmente. La etiqueta y el detalle muestran el porcentaje completo.")


CAMPOS_CASCADA = {
    "saldo_inicial": "Saldo inicial", "cobranzas": "Cobranzas", "otros_ingresos": "Otros ingresos",
    "pagos": "Pagos", "saldo_final": "Saldo final",
}


def grafico_cascada(componentes):
    filas, acumulado = [], 0
    for campo, nombre in CAMPOS_CASCADA.items():
        importe = componentes[campo]
        total = campo in {"saldo_inicial", "saldo_final"}
        inicio = 0 if total else acumulado
        fin = inicio + importe
        acumulado = fin
        signo = "" if total or importe == 0 else ("+" if importe > 0 else "−")
        filas.append({
            "concepto": nombre, "inicio": inicio / 1e6, "fin": fin / 1e6,
            "tope": max(inicio, fin) / 1e6,
            "texto": signo + dinero(abs(importe)) if not total else dinero(importe),
            "exacto": dinero(importe, abreviado=False),
            "color": AZUL if total else (VERDE if importe >= 0 else ROJO),
        })
    datos = pd.DataFrame(filas)
    datos["siguiente"] = datos["concepto"].shift(-1)
    minimo = min(0, datos[["inicio", "fin"]].min().min())
    maximo = max(0, datos[["inicio", "fin"]].max().max())
    margen = max((maximo - minimo) * 0.13, 1)
    escala = alt.Scale(domain=[minimo, maximo + margen], nice=False)
    base = alt.Chart(datos).encode(
        x=alt.X("concepto:N", title=None, sort=datos["concepto"].tolist(),
                scale=alt.Scale(paddingInner=0.32, paddingOuter=0.25),
                axis=alt.Axis(labelAngle=0, labelOverlap=False, labelExpr="split(datum.label, ' ')", labelFontSize=10)),
    )
    barras = base.mark_bar().encode(
        y=alt.Y("inicio:Q", title="Importe (millones de pesos)", scale=escala),
        y2="fin:Q", color=alt.Color("color:N", scale=None),
        tooltip=[alt.Tooltip("concepto:N", title="Concepto"), alt.Tooltip("exacto:N", title="Importe")],
    )
    conectores = base.transform_filter("isValid(datum.siguiente)").mark_rule(
        color=GRIS, opacity=0.5, strokeDash=[3, 3],
    ).encode(x2="siguiente:N", y=alt.Y("fin:Q", title="Importe (millones de pesos)", scale=escala))
    etiquetas = base.mark_text(baseline="bottom", dy=-5, color=AZUL, fontSize=10).encode(
        y=alt.Y("tope:Q", title="Importe (millones de pesos)", scale=escala), text="texto:N",
    )
    st.altair_chart(
        (conectores + barras + etiquetas).properties(height=235),
        alt="Cascada desde el saldo inicial hasta el saldo final, con cobranzas, otros ingresos y pagos",
        width="stretch",
    )


st.set_page_config(
    page_title="ExpensasAR", page_icon="🧾",
    layout="centered", initial_sidebar_state="auto",
)


try:
    resumen = cargar("resumen_mensual").set_index("mes").sort_index()
    gastos = cargar("gastos_detalle")
    gastos["rubro"] = gastos["categoria_normalizada"].fillna("Sin categoría")
    maestro = cargar("proveedores")
    unidades = cargar("unidades")
    padron = cargar("padron_uf")
    ubicaciones = padron.assign(unidad_funcional="UF_" + padron["uf_comercial"].astype(str))
    unidades = unidades.merge(
        ubicaciones[["unidad_funcional", "portal", "piso", "departamento"]],
        on="unidad_funcional", how="left", validate="many_to_one",
    )
    controles_prorrateo = cargar("validacion_prorrateo_uf")
    prorrateos = cargar("prorrateo_uf").merge(
        controles_prorrateo.loc[controles_prorrateo["estado"].eq("OK"), ["mes", "codigo_concepto"]],
        on=["mes", "codigo_concepto"], how="inner", validate="many_to_one",
    )
    liquidaciones_uf = cargar("liquidacion_uf")
    mapeo_prorrateo = cargar("mapeo_rubros_prorrateo")
    auditoria_uf = cargar("revision_reparaciones_uf")
    cascadas = cargar("caja_cascada").set_index("mes")
    morosidad = cargar("morosidad_mensual").set_index("mes").reindex(resumen.index)
    revision = cargar("movimientos_a_revisar")
    contexto_mensual = cargar("contexto_mensual").set_index("mes")
    ruta_ipc = RAIZ / "03_Notebook" / "ipc_indec_nivel_general.csv"
    ipc = leer_csv(ruta_ipc, ruta_ipc.stat().st_mtime_ns)
except (OSError, ValueError) as error:
    st.error(f"No se pudieron cargar los datos procesados. {error}")
    st.stop()

rubros, total_por_mes, cambio_rubro, contribucion = calcular_rubros(gastos, resumen.index)
proveedores_mensuales = calcular_proveedores(gastos, maestro, total_por_mes)
top_5_mes = proveedores_mensuales.groupby("mes", sort=False).head(5)
comparacion = comparar_ipc(resumen, gastos, ipc)
movimientos_uf = unidades[unidades["archivo_cargo_vinculado"].isna()]
reparaciones = movimientos_uf[movimientos_uf["tipo_movimiento"].eq("reparacion")]

with st.sidebar:
    st.header("Explorar liquidaciones")
    mes = st.selectbox("Mes", resumen.index[::-1].tolist(), format_func=lambda m: nombre_mes(m).capitalize())
    st.caption(f"{nombre_mes(resumen.index[0]).capitalize()} a {nombre_mes(resumen.index[-1])}")
    numeros_uf = padron["uf_comercial"].sort_values().tolist()
    uf_seleccion = st.selectbox(
        "Unidad funcional", ["Todas las UF", *(f"UF {numero}" for numero in numeros_uf)],
        key="filtro_uf", help="Abre Mi UF y filtra los registros documentados de la unidad.",
    )
    if uf_seleccion != "Todas las UF":
        st.markdown("[Mi UF](#mi-uf)")
    st.divider()
    st.markdown(
        "[Caja y rubros](#caja)  \n"
        "[Proveedores](#gasto) · [Gastos particulares por UF](#reparaciones)  \n"
        "[Lectura del mes](#lectura)"
    )
    st.divider()
    avanzado = st.toggle("Mostrar detalles avanzados")
fila = resumen.loc[mes]
ipc_mes = comparacion.loc[mes]
variacion = ipc_mes["variacion_total_pct"]
cambios_caja = resumen[["expensas_cobradas", "saldo_final"]].pct_change(fill_method=None).loc[mes] * 100
saldo_moroso = morosidad.loc[mes, "saldo_moroso"]
deuda_anterior = morosidad["saldo_moroso"].shift()
variacion_morosa = (morosidad["saldo_moroso"].diff().div(deuda_anterior.where(deuda_anterior.gt(0))) * 100).loc[mes]
uf_activa = uf_seleccion.replace(" ", "_") if uf_seleccion != "Todas las UF" else None
particulares_mes = movimientos_uf[movimientos_uf["mes"].eq(mes)]
pagos_vinculados_mes = unidades.loc[unidades["mes"].eq(mes) & unidades["archivo_cargo_vinculado"].notna()]
if uf_activa:
    particulares_mes = particulares_mes[particulares_mes["unidad_funcional"].eq(uf_activa)]
    pagos_vinculados_mes = pagos_vinculados_mes[pagos_vinculados_mes["unidad_funcional"].eq(uf_activa)]

st.title("ExpensasAR")
st.markdown("Entendé qué pasó con la plata de tu consorcio")
with st.container(border=True):
    st.subheader(nombre_mes(mes).capitalize())
    st.caption("Variaciones respecto del mes anterior de la serie.")
    cambios_mes = cambio_rubro.loc[mes].dropna()
    rubro_principal = cambios_mes.abs().idxmax() if not cambios_mes.empty else rubros.loc[mes].idxmax()
    ejecutivo = resumen_breve(mes, fila, variacion)
    ejecutivo = re.sub(r"(\$[\d.,]+ millones|[\d.,]+ %)", r"**\1**", ejecutivo)
    ejecutivo = ejecutivo.replace(rubro_principal.capitalize(), f"**{rubro_principal.capitalize()}**")
    st.info(ejecutivo.replace("$", r"\$"), icon=":material/insights:")

tarjetas = [
    ("Gastos Informados", dinero(fila["gastos_informados"]), variacion_destacada(variacion)),
    ("Expensas Cobradas", dinero(fila["expensas_cobradas"]), variacion_destacada(cambios_caja["expensas_cobradas"], "caja")),
    ("Saldo Final", dinero(fila["saldo_final"]), variacion_destacada(cambios_caja["saldo_final"], "caja")),
    ("Deudores Morosos", dinero(saldo_moroso), variacion_destacada(variacion_morosa)),
]
for columna, (etiqueta, valor, delta) in zip(st.columns(4, gap="small"), tarjetas):
    with columna.container(border=True, height=136, gap="small"):
        st.caption(f":blue[**{etiqueta}**]")
        mostrar_texto(f"**{valor}**")
        if delta == "Sin dato":
            st.caption("Sin comparación")
        else:
            st.markdown(delta)

if uf_activa:
    numero_uf = int(uf_activa.removeprefix("UF_"))
    ubicacion = padron.loc[padron["uf_comercial"].eq(numero_uf)].iloc[0]
    with st.container(border=True):
        st.header("Mi UF", anchor="mi-uf")
        st.subheader(f"{uf_seleccion} · Portal {ubicacion['portal']} · {ubicacion['piso']} {ubicacion['departamento']}")
        mostrar_expensa_uf(mes, numero_uf)
        st.subheader("Desglose de gastos particulares")
        st.caption("Estos cargos fueron liquidados a la unidad y forman parte del subtotal de gastos particulares.")
        cargos_uf = particulares_mes[particulares_mes["base_registro"].eq("cargo_uf")]
        for tipo, _, titulo in TIPOS_GASTOS_UF:
            cargos_tipo = cargos_uf[cargos_uf["tipo_movimiento"].eq(tipo)]
            with st.expander(f"{titulo} ({len(cargos_tipo)})"):
                if cargos_tipo.empty:
                    st.caption("Sin cargos liquidados de este tipo en el mes.")
                else:
                    tabla_gastos_uf(cargos_tipo)
        pagos_uf = particulares_mes[particulares_mes["base_registro"].eq("pago")]
        if not pagos_uf.empty:
            with st.expander("Pagos del consorcio vinculados a esta UF"):
                st.caption("Se muestran como referencia. Si el importe incluye otra unidad o un sector común, "
                           "el detalle indica el alcance del pago.")
                tabla_gastos_uf(pagos_uf)
        vinculadas = reparaciones.loc[
            reparaciones["mes"].eq(mes) & reparaciones["uf_relacionada"].eq(uf_activa)
        ]
        for reparacion in vinculadas.itertuples():
            principal = reparacion.unidad_funcional.replace("_", " ")
            st.info(f"Esta unidad figura como afectada en un pago conjunto registrado bajo **{principal}** "
                    f"({reparacion.archivo_origen}, pág. {reparacion.pagina_origen}).")
        mostrar_pagos_vinculados_uf(pagos_vinculados_mes)

with st.container(border=True):
    st.header("Qué pasó con la caja", anchor="caja")
    inicial, final = fila["saldo_inicial"], fila["saldo_final"]
    variacion_caja = round(final - inicial, 2)
    if pd.isna(variacion_caja):
        st.markdown("**Sin saldo de apertura o de cierre para comparar la caja del mes.**")
    elif variacion_caja == 0:
        st.markdown("**La caja se mantuvo estable durante el mes.**")
    else:
        direccion = "aumentó" if variacion_caja > 0 else "se redujo"
        tasa = porcentaje((final / inicial - 1) * 100, con_signo=True) if inicial != 0 else None
        cambio_pct = f" ({tasa})" if tasa is not None else ""
        mostrar_texto(f"**La caja {direccion} ${numero(abs(variacion_caja) / 1e6, 2)} M durante el mes{cambio_pct}.**")
    if pd.notna(variacion_caja):
        st.caption(f"{dinero(inicial)} → {dinero(final)}".replace("$", r"\$"))

    componentes = cascadas.loc[mes]
    if pd.isna(componentes["motivo"]):
        grafico_cascada(componentes)
        if variacion_caja < 0:
            st.write("Los egresos de efectivo superaron los ingresos del período, por lo que el consorcio utilizó parte de su saldo acumulado para cubrir la diferencia.")
        elif variacion_caja > 0:
            st.write("Los ingresos de efectivo superaron los egresos del período y el excedente se sumó al saldo acumulado del consorcio.")
        else:
            st.write("Los ingresos y los egresos de efectivo del período se compensaron.")
        st.caption(componentes["nota"])
    else:
        st.info(componentes["motivo"], icon=":material/info:")
        mostrar_texto(explicar_caja(fila))
    if mes == INICIO_RECURRENTE:
        st.caption("Mes de transición. La apertura de mayo incorpora fondos de inversión que abril mostraba por separado.")
    if pd.notna(saldo_moroso):
        mostrar_texto(f"**Morosidad al cierre: {dinero(saldo_moroso)}.**")
        st.caption("Representa recursos pendientes de cobro y puede limitar la liquidez disponible, aunque no explica por sí sola la variación de caja.")
    else:
        st.caption("Morosidad al cierre · Sin dato comparable en esta liquidación.")

    mostrar_gastos_y_prorrateo(mes, fila)

    with st.expander("Ver importes exactos del mes"):
        campos = {
            "saldo_inicial": "Saldo inicial",
            "expensas_cobradas": "Expensas cobradas",
            "otros_ingresos": "Otros ingresos informados",
            "pagos_realizados": "Pagos realizados",
            "saldo_final": "Saldo final",
            "gastos_informados": "Gastos informados",
        }
        tabla = pd.DataFrame({
            "Concepto": list(campos.values()), "Importe": [fila[campo] for campo in campos],
        })
        tabla.loc[len(tabla)] = ["Variación absoluta de caja", variacion_caja]
        tabla_monetaria(tabla, ["Importe"])

with st.container(border=True):
    st.header("Qué rubros aportaron al cambio del gasto", anchor="rubros")
    st.caption("Los principales aumentos y bajas del mes, junto con la nota de la administración.")
    if pd.isna(variacion):
        st.markdown("**Datos observados**")
        st.write("Este mes carece de una comparación con el mes anterior en la serie.")
    else:
        st.markdown(f"**Datos observados** · {variacion_destacada(variacion)} de variación mensual")
        aportes = contribucion.loc[mes].dropna()
        subas = aportes[aportes.gt(0)].nlargest(3)
        bajas = aportes[aportes.lt(0)].nsmallest(3)
        if not subas.empty:
            st.markdown(f"Mayor aporte al alza · **{subas.index[0].capitalize()}** · **{numero(subas.iloc[0])} puntos porcentuales**.")
        if not bajas.empty:
            st.markdown(f"Mayor aporte a la baja · **{bajas.index[0].capitalize()}** · **{numero(bajas.iloc[0])} puntos porcentuales**.")
        seleccion = pd.concat([subas, bajas])
        if not seleccion.empty:
            grafico = seleccion.rename("aporte").rename_axis("rubro").reset_index()
            grafico["rubro"] = grafico["rubro"].str.capitalize()
            with st.expander("Ver gráfico de aportes al cambio"):
                barras(grafico, "aporte", "Aporte al cambio total · puntos porcentuales", divergente=True)
        st.caption("Se muestran los principales aportes. El signo indica la dirección del cambio contable.")
    st.markdown("**Contexto informado por la administración**")
    st.warning(resumir_contexto(mes).replace("$", r"\$"), icon=":material/description:")

with st.container(border=True):
    st.header("Gastos frente al IPC GBA", anchor="ipc")
    modo_ipc = st.radio(
        "Período de comparación", ["Mensual", "Interanual"], horizontal=True, key="modo_ipc",
    )
    if modo_ipc == "Mensual":
        st.subheader("¿Qué gastos subieron más que la inflación este mes?")
        st.caption("Compará cuánto cambió el gasto registrado en cada rubro respecto del mes anterior con el IPC GBA.")
        columnas = st.columns(3)
        for columna, etiqueta, campo in zip(
            columnas, ["Gasto total", "Gastos habituales (sin extraordinarios)", "IPC GBA"],
            ["variacion_total_pct", "variacion_recurrente_pct", "ipc_mensual_pct"],
        ):
            with columna.container(border=True):
                st.caption(etiqueta)
                st.markdown(f"**{variacion_destacada(ipc_mes[campo], 'ipc' if campo == 'ipc_mensual_pct' else 'gasto')}**")
        st.caption("Gastos habituales excluye únicamente los rubros que la liquidación identifica expresamente como extraordinarios o no recurrentes.")
        comparables = rubros_para_ipc(rubros, mes)
        seleccion_ipc = comparables.dropna(subset=["variacion"]).head(7)
        if pd.notna(ipc_mes["ipc_mensual_pct"]) and not seleccion_ipc.empty:
            mayores = seleccion_ipc[seleccion_ipc["variacion"].gt(ipc_mes["ipc_mensual_pct"])]
            if mayores.empty:
                st.markdown("Entre los rubros del gráfico, **ninguno** registró una variación superior al IPC GBA.")
            else:
                principal = mayores.loc[mayores["variacion"].idxmax()]
                st.markdown(f"La mayor suba entre los rubros del gráfico fue en **{principal['rubro']}** · "
                            f"{variacion_destacada(principal['variacion'])}.")
        grafico_rubros_ipc(seleccion_ipc, ipc_mes["ipc_mensual_pct"])
        sin_base = comparables.head(7).loc[lambda t: t["variacion"].isna()]
        for fila_rubro in sin_base.itertuples():
            cambio = "" if pd.isna(fila_rubro.cambio) else f" · Cambio en pesos **{dinero(fila_rubro.cambio)}**"
            mostrar_texto(f"**{fila_rubro.rubro}** · Sin base comparable{cambio}")
        with st.expander("Ver todos los rubros"):
            st.caption("El gráfico prioriza hasta siete rubros con base comparable según el mayor importe entre este mes y el anterior.")
            tabla = comparables.sort_values("variacion", ascending=False).rename(columns={
                "rubro": "Rubro", "variacion": "Variación mensual", "cambio": "Cambio en pesos",
                "anterior": "Mes anterior", "actual": "Mes actual",
            })
            tabla["Variación mensual"] = tabla["Variación mensual"].map(
                lambda v: porcentaje(v, con_signo=True) if pd.notna(v) else "Sin base comparable"
            )
            tabla_monetaria(tabla[["Rubro", "Variación mensual", "Cambio en pesos", "Mes anterior", "Mes actual"]],
                            ["Cambio en pesos", "Mes anterior", "Mes actual"])
    else:
        st.subheader("Variación interanual del gasto del rubro vs IPC GBA")
        st.caption("Comparación exploratoria del crecimiento del gasto registrado frente al nivel general de precios.")
        interanuales = variaciones_interanuales(rubros, comparacion["indice_ipc"])
        opciones = interanuales.drop(columns="IPC GBA").dropna(axis=1, how="all").columns.tolist()
        if opciones:
            rubro_interanual = st.selectbox(
                "Rubro para comparar", opciones, index=opciones.index("Abonos y servicios") if "Abonos y servicios" in opciones else 0,
                format_func=str.capitalize, key="rubro_interanual",
            )
            cobertura = interanuales[[rubro_interanual, "IPC GBA"]].dropna().shape[0]
            st.caption(f"**{cobertura} de {len(interanuales)}** comparaciones disponibles entre "
                       f"{nombre_mes(interanuales.index[0])} y {nombre_mes(interanuales.index[-1])}. "
                       "La serie muestra todo el período interanual disponible.")
            grafico_interanual(interanuales, rubro_interanual)
            brechas = (interanuales[rubro_interanual] - interanuales["IPC GBA"]).dropna()
            st.caption(f"**Por encima del IPC** · {brechas.gt(1).sum()} de {cobertura} meses. "
                       f"**Por debajo del IPC** · {brechas.lt(-1).sum()} de {cobertura} meses. "
                       f"**En línea con IPC (±1 p.p.)** · {brechas.abs().le(1).sum()} de {cobertura} meses.")
            st.write("Por encima del IPC, el gasto registrado creció más que el nivel general de precios. Por debajo, creció menos. Dentro de ±1 p.p., evolucionó de manera similar. Una variación negativa indica que el gasto cayó respecto del mismo mes del año anterior.")
            st.caption("La comparación interanual reduce parte del ruido estacional al comparar cada mes con el mismo mes del año anterior, aunque no elimina cambios en cantidades, proveedores, servicios o calendario de pagos.")
            st.caption("El IPC puede verse visualmente más estable porque sus variaciones son menores que las del rubro seleccionado.")
            st.caption("Los meses sin dos importes comparables quedan sin punto. La cobertura puede variar por rubro.")
        else:
            st.info("La serie todavía no tiene importes comparables con el mismo mes del año anterior.")
    st.caption(f"[IPC GBA · INDEC]({FUENTE_IPC}), consulta del 21/11/2026.")
    st.write("La variación del gasto puede reflejar precios, cantidades, cambios en servicios o calendario de pagos.")
    if mes == INICIO_RECURRENTE:
        st.caption("La transición entre formatos cambió algunos nombres y agrupaciones de rubros. Interpretá con cautela la comparación con abril.")

with st.container(border=True):
    st.header("En qué se gastó", anchor="gasto")
    st.caption("Los rubros de mayor importe y la participación de los principales proveedores.")
    composicion = rubros.loc[mes]
    composicion = composicion[composicion.ne(0)].sort_values(ascending=False)
    grafico = composicion.rename("importe").rename_axis("rubro").reset_index()
    grafico["rubro"] = grafico["rubro"].str.capitalize()
    grafico["millones"] = grafico["importe"] / 1e6
    if not grafico.empty:
        st.markdown("**Rubros de mayor importe**")
        mostrar_texto(" · ".join(f"**{rubro.capitalize()}** ({dinero(importe)})" for rubro, importe in composicion.head(3).items()))
        with st.expander("Ver composición completa por rubro"):
            barras(grafico, "millones", "Millones de pesos")
    ranking = top_5_mes[top_5_mes["mes"].eq(mes)]
    if not ranking.empty:
        st.write(f"Proveedor principal · **{ranking.iloc[0]['nombre'].title()}**")
        columnas = st.columns(3)
        for columna, n, etiqueta in zip(columnas, [1, 3, 5], ["Proveedor principal", "Tres mayores", "Cinco mayores"]):
            columna.metric(etiqueta, porcentaje(ranking.head(n)["participacion_pct"].sum()))
        with st.expander("Principales proveedores del mes"):
            tabla = ranking[["nombre", "importe", "participacion_pct"]].rename(columns={
                "nombre": "Proveedor", "importe": "Importe", "participacion_pct": "Participación",
            })
            tabla["Participación"] = tabla["Participación"].map(porcentaje)
            tabla_monetaria(tabla, ["Importe"])
    else:
        st.write("No hay proveedores atribuibles en este mes.")
    ranking_extra = concentracion_trabajos_extraordinarios(gastos)
    periodo = f"{nombre_mes(resumen.index[0])} a {nombre_mes(resumen.index[-1])}"
    with st.expander("Concentración de proveedores en trabajos extraordinarios"):
        if ranking_extra.empty:
            st.write("No se encontraron movimientos vinculados a trabajos extraordinarios en la serie.")
        else:
            st.caption(f"Serie completa · {periodo}. Los movimientos son pagos registrados y un mismo trabajo puede reunir varios.")
            mostrar_texto(
                f"**{ranking_extra['Movimientos'].sum()} movimientos** por un total de "
                f"**{dinero(ranking_extra['Importe'].sum(), abreviado=False)}**."
            )
            tabla_importes = ranking_extra[["Proveedor", "Movimientos", "Importe", "Participación"]].copy()
            tabla_importes["Participación"] = tabla_importes["Participación"].map(lambda v: f"{numero(v, 2)} %")
            tabla_monetaria(tabla_importes, ["Importe"])
            st.caption("Una concentración elevada no implica por sí misma una irregularidad. Este indicador muestra qué proveedores tuvieron mayor peso dentro de los trabajos extraordinarios registrados.")
    if not ranking_extra.empty:
        st.subheader("Proveedores recurrentes")
        st.caption(f"Trabajos extraordinarios y no recurrentes · {periodo}.")
        st.write("El importe acumulado refleja el volumen de pagos. Los meses activos muestran con qué frecuencia aparece cada proveedor a lo largo de la serie.")
        st.caption("La recurrencia muestra cuántos meses distintos registraron pagos a un proveedor. Una frecuencia alta no implica por sí misma una irregularidad.")
        with st.expander("Ver ranking por meses activos"):
            por_meses = ranking_extra.sort_values(
                ["meses_activos", "Importe", "Proveedor"], ascending=[False, False, True],
            )
            tabla_rec = por_meses[[
                "Proveedor", "meses_activos", "Movimientos", "Importe", "promedio_activo",
                "primer_mes", "ultimo_mes",
            ]].rename(columns={
                "meses_activos": "Meses activos", "Importe": "Importe acumulado",
                "promedio_activo": "Promedio por mes activo", "primer_mes": "Primer mes",
                "ultimo_mes": "Último mes",
            })
            tabla_monetaria(tabla_rec, ["Importe acumulado", "Promedio por mes activo"])


with st.container(border=True):
    st.header("Gastos particulares por UF", anchor="reparaciones")
    st.caption("Movimientos del mes vinculados a una unidad. Los importes incluyen cargos liquidados y pagos del consorcio, identificados por separado en el detalle.")
    st.markdown(f"**{len(particulares_mes)} movimientos** vinculados a **{particulares_mes['unidad_funcional'].nunique()} UF**.")
    if particulares_mes["alcance_importe"].notna().any():
        st.caption("Algunos pagos también cubren otra unidad o un sector común. Se conserva el importe completo y su alcance.")
    por_tipo = particulares_mes.groupby("tipo_movimiento").agg(
        Movimientos=("importe", "size"), UF=("unidad_funcional", "nunique"), Importe=("importe", "sum"),
    )
    resumen_uf = por_tipo.reindex([tipo for tipo, _, _ in TIPOS_GASTOS_UF], fill_value=0).reset_index(drop=True)
    resumen_uf.insert(0, "Tipo", [etiqueta for _, etiqueta, _ in TIPOS_GASTOS_UF])
    tabla_monetaria(resumen_uf, ["Importe"])
    if uf_activa:
        st.markdown("[Ver el detalle de esta unidad en Mi UF](#mi-uf)")
    else:
        for tipo, titulo, _ in TIPOS_GASTOS_UF:
            movimientos = particulares_mes[particulares_mes["tipo_movimiento"].eq(tipo)]
            with st.expander(titulo):
                if movimientos.empty:
                    st.caption("Sin movimientos documentados de este tipo en el mes.")
                else:
                    tabla_gastos_uf(movimientos)
        mostrar_pagos_vinculados_uf(pagos_vinculados_mes)
        ambiguas = auditoria_uf.loc[auditoria_uf["mes"].eq(mes) & auditoria_uf["estado"].eq("unidad probable pero ambigua")]
        if not ambiguas.empty:
            with st.expander(f"Ver referencias ambiguas ({len(ambiguas)})"):
                st.caption("Estos importes quedan fuera del total asignado a unidades.")
                st.dataframe(ambiguas[["referencia_pdf", "normalizacion", "pagina_origen"]].rename(columns={
                    "referencia_pdf": "Referencia del PDF", "normalizacion": "Identificación", "pagina_origen": "Página",
                }), hide_index=True, width="stretch")

with st.container(border=True):
    st.header("Movimientos que requieren explicación adicional", anchor="revisiones")
    st.caption("Casos documentales del mes y su diferencia absoluta cuando está disponible.")
    casos = revision[revision["mes"].eq(mes)]
    columna_casos, columna_diferencia = st.columns(2)
    columna_casos.metric("Movimientos del mes", len(casos))
    diferencias = casos["diferencia"].dropna().abs()
    if not diferencias.empty:
        columna_diferencia.metric("Diferencia absoluta total", dinero(diferencias.sum(), abreviado=False))
    if not casos.empty:
        with st.expander("Ver los movimientos del mes"):
            tabla = casos[["proveedor", "concepto", "importe_factura", "importe_pagado", "diferencia", "motivo_revision"]].rename(columns={
                "proveedor": "Proveedor", "concepto": "Concepto", "importe_factura": "Importe factura",
                "importe_pagado": "Importe pagado", "diferencia": "Diferencia", "motivo_revision": "Motivo de revisión",
            })
            tabla_monetaria(tabla, ["Importe factura", "Importe pagado", "Diferencia"])
    st.caption("Estos movimientos no implican error, fraude ni sobreprecio. Solo presentan diferencias documentales que merecen revisión.")

with st.container(border=True):
    st.header("Lectura automática del mes", anchor="lectura")
    bloques_del_mes = [bloque.split("\n", 1) for bloque in generar_resumen_mes(mes).split("\n\n")]
    estilos = (
        (st.info, ":material/insights:"),
        (st.warning, ":material/description:"),
        (st.success, ":material/summarize:"),
    )
    for (titulo, texto), (mostrar, icono) in zip(bloques_del_mes, estilos):
        st.markdown(f"**{titulo.rstrip(':')}**")
        mostrar(texto.replace("$", r"\$"), icon=icono)

if avanzado:
    with st.container(border=True):
        st.header("Detalles avanzados")
        estadistica = calcular_alertas(gastos, resumen, comparacion["indice_ipc"])
        estadistica = estadistica[estadistica["mes"].eq(mes)].copy()
        estadistica["Alerta"] = estadistica["z_score"].abs().ge(2.5)
        with st.expander("Alertas estadísticas y gasto real"):
            st.caption("Pesos de marzo de 2025. La alerta usa |z| ≥ 2,5 y los seis meses anteriores comparables. Inusual significa alejado del patrón reciente.")
            st.write(f"Alertas estadísticas del mes · {estadistica['Alerta'].sum()}")
            estadistica = estadistica.rename(columns={"rubro": "Rubro", "gasto_real": "Gasto real", "z_score": "z-score"})
            estadistica["z-score"] = estadistica["z-score"].round(2)
            tabla_monetaria(estadistica[["Rubro", "Gasto real", "z-score", "Alerta"]], ["Gasto real"])

        with st.expander("Todos los proveedores del mes"):
            todos = proveedores_mensuales[proveedores_mensuales["mes"].eq(mes)]
            tabla = todos[["nombre", "importe", "categoria_predominante", "participacion_pct"]].rename(columns={
                "nombre": "Proveedor", "importe": "Importe", "categoria_predominante": "Categoría predominante histórica",
                "participacion_pct": "Participación",
            })
            tabla["Participación"] = tabla["Participación"].map(porcentaje)
            tabla_monetaria(tabla, ["Importe"])

