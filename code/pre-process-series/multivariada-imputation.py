import gc
import os
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd


links = {
    "100": [
        "ap-am", "am-ap", "ap-pa", "pa-ap", "pa-to", "to-pa", "to-df", "df-to",
        "df-mt", "mt-df", "mt-ro", "ro-mt", "mt-ms", "ms-mt", "df-ma", "ma-df",
        "ma-pi", "pi-ma", "pi-ba", "ba-pi", "df-rj", "rj-df", "rj-es", "es-rj",
        "ba-df", "df-ba", "pi-ce", "ce-pi", "ce-rn", "rn-ce", "rn-pb", "pb-rn",
        "pb-pe", "pe-pb", "pe-al", "al-pe", "al-se", "se-al", "se-ba", "ba-se",
        "ce-ba", "ba-ce", "ba-mg", "mg-ba", "ba-es", "es-ba", "mg-sp", "sp-mg",
        "sp-rs", "rs-sp", "pr-sc", "sc-pr", "rs-sc", "sc-rs",
    ],
    "200": ["ce-sp", "sp-ce", "rj-sp", "sp-rj", "pr-rs", "rs-pr"],
    "300": ["pr-sp", "sp-pr"],
    "40": ["pb-ba", "ba-pb"],
    "20": ["go-df", "df-go", "df-sp", "sp-df", "mg-rj", "rj-mg"],
    "6": ["ac-ro", "ro-ac"],
    "3": ["am-df", "df-am"],
    "1": ["am-rr", "rr-am", "rr-ce", "ce-rr"],
    "10": [
        "mt-go", "go-mt", "ms-go", "go-ms", "ms-pr", "pr-ms", "df-mg", "mg-df",
        "mg-es", "es-mg", "pa-ma", "ma-pa", "df-ce", "ce-df",
    ],
}


def contar_ocorrencias(row):
    return row.astype(str).str.contains("bkb.rnp.br|No Hostname|pop").sum()


def substituir_prefixo(df):
    def processar_celula(valor):
        if isinstance(valor, str) and ".bkb.rnp.br" in valor:
            prefixo = valor.split(".")[0].split("-")
            prefixo = prefixo[0] + "-" + prefixo[1]
            prefixo = re.sub(r"\d+|lan|mx", "", prefixo)
            partes = prefixo.split("-")
            if len(partes) > 0 and len(partes[0]) == 3:
                partes[0] = partes[0][1:]
            if len(partes) > 1 and len(partes[1]) == 3:
                partes[1] = partes[1][1:]
            return "-".join(partes)
        return np.nan

    colunas = df.columns.difference(
        ["Timestamp", "Data", "Hop_count", "Bottleneck", "Link_bottleneck"]
    )
    df[colunas] = df[colunas].map(processar_celula)
    return df


def obter_menor_chave(linha, links):
    chaves_encontradas, links_gargalo = [], []
    for link in linha:
        for chave, valores in links.items():
            if link in valores:
                chaves_encontradas.append(int(chave))
                links_gargalo.append(link)
    if chaves_encontradas:
        idx = np.argmin(chaves_encontradas)
        return chaves_encontradas[idx], links_gargalo[idx]
    return np.nan, np.nan


def aplicar_bottleneck(df, links):
    retorno = df.apply(
        lambda linha: obter_menor_chave(linha, links), axis=1, result_type="expand"
    )
    df[["Bottleneck", "Link_bottleneck"]] = retorno
    return df


def processar_traceroute_df(df):
    """Processa DataFrame de traceroute já lido"""
    max_cols = len(df.columns)
    if max_cols < 2:
        raise ValueError("DataFrame não tem colunas suficientes")

    df.columns = ['Timestamp', 'Data'] + [f'Hop_{i}' for i in range(1, max_cols - 1)]
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.dropna(subset=['Data'])

    df['Hop_count'] = df.iloc[:, 2:].apply(contar_ocorrencias, axis=1)

    df = substituir_prefixo(df)
    df = aplicar_bottleneck(df, links)

    df_final = df[['Timestamp', 'Data', 'Hop_count', 'Bottleneck', 'Link_bottleneck']]
    df_final = df_final.set_index('Data').resample('10min').agg({
        'Timestamp': 'first',
        'Hop_count': 'mean',
        'Bottleneck': 'min',
        'Link_bottleneck': lambda x: x.mode()[0] if not x.mode().empty else np.nan
    }).reset_index()

    df_final['Timestamp'] = pd.to_datetime(df_final['Data']).astype('int64') // 10**9
    df_final['Hop_count'] = df_final['Hop_count'].round(2)
    df_final = df_final.dropna(subset=['Hop_count', 'Link_bottleneck'])

    return df_final


def processar_atraso_df(df):
    """Processa DataFrame de atraso - reamostra de 1 em 1 minuto"""
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.dropna(subset=['Data'])
    df = df.set_index('Data').resample('1min').mean().reset_index()
    df['Timestamp'] = df['Data'].astype('int64') // 10**9
    df['Data'] = df['Data'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['Atraso(ms)'] = df['Atraso(ms)'].round(2)
    df = df.dropna(subset=['Atraso(ms)'])
    return df[['Timestamp', 'Data', 'Atraso(ms)']]


def processar_vazao_df(df):
    """Processa DataFrame de vazão - reamostra de 6 em 6 horas"""
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.dropna(subset=['Data'])
    df = df.set_index('Data').resample('6h').mean().reset_index()
    df['Timestamp'] = df['Data'].astype('int64') // 10**9
    df['Data'] = df['Data'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['Vazao'] = pd.to_numeric(df['Vazao'], errors='coerce').round(2)
    df = df.dropna(subset=['Vazao'])
    return df[['Timestamp', 'Data', 'Vazao']]


def extrair_substring(nome_arquivo):
    padrao = re.compile(r"([a-z]{2}-[a-z]{2})")
    match = padrao.search(nome_arquivo)
    return match.group(1) if match else None


def merge_dataframes(
    df_vazao_bbr, df_vazao_cubic, df_atraso, df_traceroute, intervalo_maximo="10min"
):
    """Merge usando vazão BBR como base (6h), buscando atraso e traceroute mais próximos"""
    for df in [df_vazao_bbr, df_vazao_cubic, df_atraso, df_traceroute]:
        df["Timestamp"] = pd.to_datetime(
            pd.to_numeric(df["Timestamp"], errors="coerce"), unit="s"
        )
        df.sort_values("Timestamp", inplace=True)

    # Usar vazão BBR como base (intervalo de 6h)
    merged = df_vazao_bbr.copy()
    
    # Fazer merge com atraso (buscar o mais próximo dentro de 3h)
    merged = pd.merge_asof(
        merged,
        df_atraso,
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(intervalo_maximo),
    )
    
    # Fazer merge com traceroute (buscar o mais próximo dentro de 3h)
    merged = pd.merge_asof(
        merged,
        df_traceroute,
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(intervalo_maximo),
    )
    
    # Fazer merge com vazão CUBIC (deve ter os mesmos timestamps)
    merged = pd.merge_asof(
        merged,
        df_vazao_cubic,
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(intervalo_maximo),
    )
    
    return merged.loc[:, ~merged.columns.duplicated()]


def main():
    # Caminhos base
    pasta_results = os.path.join("..", "..", "datasets", "multivariada-series", "post-processed")
    pasta_antiga_base = os.path.join(
        "..", "..", "datasets", "datasets-todos-links", "datasets-2024", "originals"
    )
    pasta_2025_base = os.path.join(
        "..", "..", "datasets", "datasets-todos-links", "datasets-2025", "originals"
    )
    pasta_2025_2_base = os.path.join(
        "..", "..", "datasets", "datasets-todos-links", "datasets-2025-2", "originals"
    )

    def build_paths(base):
        return {
            "atraso": os.path.join(base, "datasets atraso"),
            "traceroute": os.path.join(base, "datasets traceroute"),
            "vazao_bbr": os.path.join(base, "datasets vazao", "bbr"),
            "vazao_cubic": os.path.join(base, "datasets vazao", "cubic"),
        }

    paths_2024 = build_paths(pasta_antiga_base)
    paths_2025 = build_paths(pasta_2025_base)
    paths_2025_2 = build_paths(pasta_2025_2_base)

    pasta_saida = os.path.join(pasta_results, "multivariada-mean")
    Path(pasta_saida).mkdir(parents=True, exist_ok=True)

    def coletar_todos_arquivos(pastas, substring, prefixo=None):
        caminhos = []
        for pasta in pastas:
            for arq in os.listdir(pasta):
                if arq.endswith(".csv") and substring in arq:
                    if prefixo:
                        if arq.lower().startswith(prefixo):
                            caminhos.append(os.path.join(pasta, arq))
                    else:
                        caminhos.append(os.path.join(pasta, arq))
        return caminhos

    def obter_substrings_atuais(pastas, prefixo):
        substrings = set()
        for pasta in pastas:
            for arq in os.listdir(pasta):
                if arq.endswith(".csv") and arq.lower().startswith(prefixo):
                    substring = extrair_substring(arq)
                    if substring:
                        substrings.add(substring)
        return substrings

    substrings_bbr = obter_substrings_atuais(
        [paths_2024["vazao_bbr"], paths_2025["vazao_bbr"], paths_2025_2["vazao_bbr"]],
        "bbr",
    )
    substrings_cubic = obter_substrings_atuais(
        [
            paths_2024["vazao_cubic"],
            paths_2025["vazao_cubic"],
            paths_2025_2["vazao_cubic"],
        ],
        "cubic",
    )
    substrings_em_comum = substrings_bbr.intersection(substrings_cubic)

    for substring in sorted(substrings_em_comum):
        print(f"Processando: {substring}")
        try:
            caminhos_bbr = coletar_todos_arquivos(
                [
                    paths_2024["vazao_bbr"],
                    paths_2025["vazao_bbr"],
                    paths_2025_2["vazao_bbr"],
                ],
                substring,
                prefixo="bbr",
            )
            caminhos_cubic = coletar_todos_arquivos(
                [
                    paths_2024["vazao_cubic"],
                    paths_2025["vazao_cubic"],
                    paths_2025_2["vazao_cubic"],
                ],
                substring,
                prefixo="cubic",
            )
            if not caminhos_bbr or not caminhos_cubic:
                print(
                    f"Arquivos de vazão BBR ou CUBIC faltando para {substring}, pulando..."
                )
                continue

            # Processar vazão BBR - 6 em 6 horas
            df_vazao_bbr_raw = pd.concat(
                [pd.read_csv(p) for p in caminhos_bbr]
            ).drop_duplicates()
            df_vazao_bbr = processar_vazao_df(df_vazao_bbr_raw).sort_values("Timestamp")
            df_vazao_bbr = df_vazao_bbr.rename(
                columns={"Data": "Data_vazao_bbr", "Vazao": "Vazao_BBR"}
            )

            # Processar vazão CUBIC - 6 em 6 horas
            df_vazao_cubic_raw = pd.concat(
                [pd.read_csv(p) for p in caminhos_cubic]
            ).drop_duplicates()
            df_vazao_cubic = processar_vazao_df(df_vazao_cubic_raw).sort_values(
                "Timestamp"
            )
            df_vazao_cubic = df_vazao_cubic.rename(
                columns={"Data": "Data_vazao_cubic", "Vazao": "Vazao_CUBIC"}
            )

            caminhos_atraso = coletar_todos_arquivos(
                [paths_2024["atraso"], paths_2025["atraso"], paths_2025_2["atraso"]],
                substring,
            )
            caminhos_traceroute = coletar_todos_arquivos(
                [
                    paths_2024["traceroute"],
                    paths_2025["traceroute"],
                    paths_2025_2["traceroute"],
                ],
                substring,
            )

            if not caminhos_atraso or not caminhos_traceroute:
                print(
                    f"Arquivos de atraso/traceroute faltando para {substring}, pulando..."
                )
                continue

            # Processar atraso - 1 em 1 minuto
            df_atraso_raw = pd.concat(
                [pd.read_csv(c) for c in caminhos_atraso]
            ).drop_duplicates()
            df_atraso = processar_atraso_df(df_atraso_raw).sort_values("Timestamp")

            # Processar traceroute - 10 em 10 minutos
            linhas = []
            for c in caminhos_traceroute:
                with open(c, "r") as f:
                    linhas.extend(f.readlines())
            linhas_unicas = list(set(l.strip() for l in linhas if l.strip()))
            df_traceroute_raw = pd.DataFrame(
                [linha.split(",") for linha in linhas_unicas]
            )
            df_traceroute = processar_traceroute_df(df_traceroute_raw).sort_values(
                "Timestamp"
            )

            df_atraso = df_atraso.rename(columns={"Data": "Data_atraso"})
            df_traceroute = df_traceroute.rename(columns={"Data": "Data_traceroute"})

            df_merged = merge_dataframes(
                df_vazao_bbr, df_vazao_cubic, df_atraso, df_traceroute
            )
            df_merged = df_merged[
                [
                    "Data_vazao_bbr",
                    "Atraso(ms)",
                    "Hop_count",
                    "Bottleneck",
                    "Vazao_BBR",
                ]
            ]
            df_merged = df_merged.rename(columns={"Data_vazao_bbr": "Data"})
            df_merged["Data"] = pd.to_datetime(df_merged["Data"])
            df_merged = df_merged.dropna(subset=["Data"])
            df_merged = df_merged.sort_values("Data").reset_index(drop=True)

            # Criar série temporal contínua de 6 em 6 horas
            if len(df_merged) > 0:
                data_inicio = df_merged["Data"].min()
                data_fim = df_merged["Data"].max()
                
                # Criar range completo de 6 em 6 horas
                date_range_completo = pd.date_range(
                    start=data_inicio, 
                    end=data_fim, 
                    freq='6h'
                )
                
                # Criar DataFrame com todas as datas
                df_completo = pd.DataFrame({'Data': date_range_completo})
                
                # Fazer merge com os dados existentes
                df_merged = pd.merge(df_completo, df_merged, on='Data', how='left')
                
                # Calcular medianas antes do preenchimento (apenas dos valores não-nulos)
                mediana_atraso = df_merged["Atraso(ms)"].median()
                mediana_hop_count = df_merged["Hop_count"].median()
                mediana_bottleneck = df_merged["Bottleneck"].median()
                
                # Preencher valores faltantes
                df_merged["Atraso(ms)"] = df_merged["Atraso(ms)"].fillna(mediana_atraso)
                df_merged["Hop_count"] = df_merged["Hop_count"].fillna(mediana_hop_count)
                df_merged["Bottleneck"] = df_merged["Bottleneck"].fillna(mediana_bottleneck)
                df_merged["Vazao_BBR"] = df_merged["Vazao_BBR"].fillna(-1)
                
                print(
                    f"Intervalo contínuo de {substring}: {df_merged['Data'].min()} até {df_merged['Data'].max()}"
                )
                print(f"Total de registros: {len(df_merged)} (série contínua de 6h)")
            else:
                print(f"Nenhum dado disponível para {substring}")
                continue

            if len(df_merged) > 10:
                df_merged.to_csv(
                    os.path.join(pasta_saida, f"{substring}_merged.csv"), index=False
                )
                print(f"Salvo: {substring}_merged.csv")
                gc.collect()
            else:
                print(f"Link {substring} tem menos de 10 amostras.")
        except Exception as e:
            print(f"Erro ao processar {substring}: {e}")


if __name__ == "__main__":
    inicio = time.time()
    main()
    fim = time.time()
    duracao = fim - inicio
    print(f"Tempo total de execução: {duracao:.2f} segundos")