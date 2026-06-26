import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 中文乱码解决
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
st.set_page_config(layout="wide", page_title="浙江省经济可视化分析")
st.title("📊 浙江省经济社会发展多维可视化分析系统")

# ========== 1. 数据加载与清洗 ==========
def clean_year_columns(df):
    cols = list(df.columns)
    new_cols = [cols[0]]
    for col in cols[1:]:
        if isinstance(col, str):
            col_clean = col.strip().replace('年', '').replace(' ', '')
            try:
                new_cols.append(int(col_clean))
            except ValueError:
                new_cols.append(col)
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df

@st.cache_data
def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8", sep=",")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk", sep=",")

@st.cache_data
def load_all_data():
    df_gdp_zj = read_csv_safe("ZJ-GDP年度分区数据.csv")
    df_pop = read_csv_safe("ZJ-年度人口数据.csv")
    df_emp = read_csv_safe("ZJ-年度就业数据.csv")
    df_unemp = read_csv_safe("ZJ-年度失业数据.csv")
    df_gdp_prov = read_csv_safe("GDP分省年度数据.csv")
    df_gdp_zj = clean_year_columns(df_gdp_zj)
    df_pop = clean_year_columns(df_pop)
    df_emp = clean_year_columns(df_emp)
    df_unemp = clean_year_columns(df_unemp)
    df_gdp_prov = clean_year_columns(df_gdp_prov)
    return df_gdp_zj, df_pop, df_emp, df_unemp, df_gdp_prov

df_gdp_zj, df_pop, df_emp, df_unemp, df_gdp_prov = load_all_data()

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("🔍 数据筛选面板")
    year_list = [col for col in df_gdp_zj.columns[1:] if isinstance(col, int)]
    if len(year_list) == 0:
        year_list = list(range(2000, 2023))
    min_y, max_y = st.slider("年份范围", min(year_list), max(year_list), (min(year_list), max(year_list)))
    indicators = st.multiselect("选择分析指标", ["GDP", "人口", "就业", "失业"], default=["GDP", "人口", "就业", "失业"])

    st.divider()
    st.subheader("📌 图表简化控制")
    # 相关性指标选择
    corr_vars = st.multiselect(
        "相关性分析指标（建议≤5个）",
        ["GDP总值", "年末常住人口", "城镇单位就业人员", "城镇登记失业人数", "城镇登记失业率"],
        default=["GDP总值", "年末常住人口", "城镇单位就业人员", "城镇登记失业率"]
    )
    # 地区显示数量
    top_n = st.slider("地区GDP显示数量（柱状图）", 5, 30, 10)
    pie_n = st.slider("饼图显示最大切片数", 3, 15, 8)

# ========== 2. 数据概览 ==========
st.subheader("1. 数据集概览")
tab1, tab2, tab3, tab4 = st.tabs(["浙江分区GDP", "浙江人口", "浙江就业", "浙江失业"])
with tab1:
    st.dataframe(df_gdp_zj, use_container_width=True)
with tab2:
    st.dataframe(df_pop, use_container_width=True)
with tab3:
    st.dataframe(df_emp, use_container_width=True)
with tab4:
    st.dataframe(df_unemp, use_container_width=True)

# ========== 3. 时序趋势 ==========
st.subheader("2. 浙江省经济指标时序趋势")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### GDP年度变化趋势")
    try:
        gdp_row = df_gdp_zj[df_gdp_zj.iloc[:, 0].str.contains("浙江省_地区生产总值", na=False)]
        if not gdp_row.empty:
            gdp_series = gdp_row.iloc[0, 1:]
            year_cols = [col for col in df_gdp_zj.columns[1:] if isinstance(col, int)]
            valid_years = year_cols[:len(gdp_series)]
            gdp_df = pd.DataFrame({"年份": valid_years, "GDP总值": gdp_series.values[:len(valid_years)]})
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=gdp_df, x="年份", y="GDP总值", marker="o", ax=ax1)
            plt.xticks(rotation=45)
            st.pyplot(fig1)
        else:
            st.info("未找到浙江省GDP数据行")
    except Exception as e:
        st.info(f"GDP绘图异常：{e}")

with col2:
    st.markdown("#### 总人口变化趋势")
    try:
        df_pop_t = df_pop.set_index(df_pop.columns[0]).T.reset_index()
        df_pop_t.columns = ["年份"] + df_pop_t.columns[1:].tolist()
        pop_col = df_pop_t.columns[1]
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=df_pop_t, x="年份", y=pop_col, marker="s", color="orange", ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)
    except Exception as e:
        st.info(f"人口绘图异常：{e}")

# ========== 4. 相关性分析（简化版） ==========
st.subheader("3. 经济指标相关性分析")
try:
    # 准备合并数据
    gdp_row = df_gdp_zj[df_gdp_zj.iloc[:, 0].str.contains("浙江省_地区生产总值", na=False)]
    if not gdp_row.empty:
        gdp_series = gdp_row.iloc[0, 1:]
        year_cols = [col for col in df_gdp_zj.columns[1:] if isinstance(col, int)]
        valid_years = year_cols[:len(gdp_series)]
        df_gdp_t = pd.DataFrame({"年份": valid_years, "GDP总值": gdp_series.values[:len(valid_years)]})
    else:
        raise ValueError("GDP数据缺失")

    df_pop_t = df_pop.set_index(df_pop.columns[0]).T.reset_index()
    df_pop_t.columns = ["年份"] + df_pop_t.columns[1:].tolist()
    df_emp_t = df_emp.set_index(df_emp.columns[0]).T.reset_index()
    df_emp_t.columns = ["年份"] + df_emp_t.columns[1:].tolist()
    df_unemp_t = df_unemp.set_index(df_unemp.columns[0]).T.reset_index()
    df_unemp_t.columns = ["年份"] + df_unemp_t.columns[1:].tolist()

    merged = df_gdp_t.merge(df_pop_t, on="年份", how="inner")
    merged = merged.merge(df_emp_t, on="年份", how="inner")
    merged = merged.merge(df_unemp_t, on="年份", how="inner")
    merged = merged.dropna()

    # 从合并数据中提取用户选择的相关性指标
    # 构建一个列名映射（中文关键字 → 实际列名）
    col_mapping = {
        "GDP总值": "GDP总值",
        "年末常住人口": [c for c in merged.columns if "年末常住人口" in c][0] if any("年末常住人口" in c for c in merged.columns) else None,
        "城镇单位就业人员": [c for c in merged.columns if "城镇单位就业人员" in c][0] if any("城镇单位就业人员" in c for c in merged.columns) else None,
        "城镇登记失业人数": [c for c in merged.columns if "城镇登记失业人数" in c][0] if any("城镇登记失业人数" in c for c in merged.columns) else None,
        "城镇登记失业率": [c for c in merged.columns if "城镇登记失业率" in c][0] if any("城镇登记失业率" in c for c in merged.columns) else None,
    }
    # 过滤掉None
    selected_cols = [col_mapping[v] for v in corr_vars if col_mapping.get(v) is not None]
    if len(selected_cols) >= 2:
        corr_df = merged[selected_cols].dropna()
        fig_corr, ax_c = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_df.corr(), annot=True, cmap="RdBu_r", center=0, ax=ax_c)
        st.pyplot(fig_corr)
    else:
        st.info("请至少选择2个指标进行相关性分析")
except Exception as e:
    st.info(f"相关性计算失败：{e}")

# ========== 5. 分区GDP对比（简化版） ==========
st.subheader("4. 浙江省各地区GDP对比")
col3, col4 = st.columns(2)

with col3:
    st.markdown(f"#### 各地区GDP柱状图（2020年，前{top_n}名）")
    try:
        reg_col = df_gdp_zj.columns[0]
        target_year = 2020
        if target_year in df_gdp_zj.columns:
            plot_data = df_gdp_zj[[reg_col, target_year]].dropna()
            # 按GDP降序排列，取前top_n
            plot_data_sorted = plot_data.sort_values(by=target_year, ascending=False).head(top_n)
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plot_data_sorted, x=reg_col, y=target_year, palette="viridis", ax=ax3)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel("地区")
            plt.ylabel("GDP（亿元）")
            st.pyplot(fig3)
        else:
            st.info("2020年数据不存在")
    except Exception as e:
        st.info(f"柱状图异常：{e}")

with col4:
    st.markdown(f"#### 各地区2020年GDP占比饼图（前{pie_n}名 + 其他）")
    try:
        reg_col = df_gdp_zj.columns[0]
        target_year = 2020
        if target_year in df_gdp_zj.columns:
            plot_data = df_gdp_zj[[reg_col, target_year]].dropna()
            # 按GDP降序排列
            plot_data_sorted = plot_data.sort_values(by=target_year, ascending=False)
            # 取前pie_n-1个单独显示，其余合并为“其他”
            if len(plot_data_sorted) > pie_n:
                top_data = plot_data_sorted.iloc[:pie_n-1]
                others_sum = plot_data_sorted.iloc[pie_n-1:][target_year].sum()
                other_row = pd.DataFrame({reg_col: ["其他"], target_year: [others_sum]})
                pie_data = pd.concat([top_data, other_row], ignore_index=True)
            else:
                pie_data = plot_data_sorted
            fig_pie = px.pie(pie_data, values=target_year, names=reg_col)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("2020年数据不存在")
    except Exception as e:
        st.info(f"饼图异常：{e}")

# ========== 6. 平行坐标 ==========
st.subheader("6. 多维指标平行坐标图")
try:
    if 'merged' in locals() and not merged.empty:
        num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            fig_pc = px.parallel_coordinates(merged, dimensions=num_cols, color=num_cols[0])
            st.plotly_chart(fig_pc, use_container_width=True)
        else:
            st.info("数据维度不足")
    else:
        st.info("请先运行相关性分析生成合并数据")
except Exception as e:
    st.info(f"平行坐标加载失败：{e}")

# ========== 7. PCA ==========
st.subheader("7. 经济指标PCA主成分分析")
try:
    if 'merged' in locals() and not merged.empty:
        features = merged.drop("年份", axis=1, errors="ignore")
        features = features.select_dtypes(include=[np.number]).dropna()
        if features.shape[1] >= 2:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)
            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(scaled)

            st.write(f"PC1方差解释率：{pca.explained_variance_ratio_[0]:.2%}")
            st.write(f"PC2方差解释率：{pca.explained_variance_ratio_[1]:.2%}")
            st.write(f"累计方差解释率：{sum(pca.explained_variance_ratio_):.2%}")

            pca_df = pd.DataFrame(pca_res, columns=["PC1", "PC2"])
            pca_df["年份"] = merged["年份"]
            if "GDP总值" in merged.columns:
                pca_df["GDP总值"] = merged["GDP总值"]
                fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="GDP总值", hover_data=["年份"],
                                     title="经济指标PCA二维降维散点图")
            else:
                fig_pca = px.scatter(pca_df, x="PC1", y="PC2", hover_data=["年份"],
                                     title="经济指标PCA二维降维散点图")
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("特征数量不足（至少需要2个）")
    else:
        st.info("请先运行相关性分析生成合并数据")
except Exception as e:
    st.info(f"PCA计算失败：{e}")

st.success("✅ 浙江省经济社会发展多维可视化分析系统运行完成！")