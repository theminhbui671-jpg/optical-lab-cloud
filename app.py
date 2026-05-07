import os

# ==========================================
# 0. 云端兼容设置
# ==========================================
# 云端部署时不要强制清空代理；仅在本地需要时通过环境变量启用。
if os.getenv("DISABLE_PROXY", "0") == "1":
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["all_proxy"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import random

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    OPENAI_IMPORT_ERROR = ""
except ModuleNotFoundError as exc:
    OpenAI = None
    OPENAI_AVAILABLE = False
    OPENAI_IMPORT_ERROR = str(exc)

# ==========================================
# ✅【新增】后端算法库引用 (Phase 1 核心)
# ==========================================
import sys
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from IFTS.simulation_main.modul_main import tx_main, channel_main, rx_main, sig_main
    from IFTS.simulation_main.modul_para import simulation_para, signal_para, txsignal_para, channel_para, rxsignal_para, sigplot_para
    IFTS_AVAILABLE = True
except ImportError as e:
    IFTS_AVAILABLE = False
    print(f"IFTS导入失败: {e}")
# ==========================================

AI_CLIENT_ERROR = ""
DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LM_STUDIO_API_KEY = "lm-studio"
DEFAULT_MODEL_FALLBACK = "repository@q4_k_m"

def get_secret_or_env(name, default=""):
    """优先读取 Streamlit Secrets，失败时回退到环境变量。"""
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

def show_ai_unavailable(feature_name):
    reason = AI_CLIENT_ERROR or "当前处于规则库模式或未配置在线/本地大模型。"
    st.warning(f"{feature_name} 当前使用云端兼容降级：{reason}")
    st.info("云端教学版可正常完成所有仿真实验；需要大模型时，请在 Streamlit Secrets 中配置 OPENAI_API_KEY/OPENAI_BASE_URL，或本地运行时选择 LM Studio。")

def evaluate_link_quality(metrics):
    ber = metrics.get("ber", 1.0)
    q = metrics.get("q_factor", 0.0)
    osnr = metrics.get("osnr", 0.0)
    rx_power = metrics.get("rx_power", -100.0)

    score = 100
    notes = []
    if ber > 1e-3:
        score -= 35; notes.append("BER 高于 1e-3，链路可靠性较差。")
    elif ber > 1e-6:
        score -= 18; notes.append("BER 处于临界区，建议优化 OSNR 或缩短距离。")
    elif ber > 1e-9:
        score -= 8; notes.append("BER 基本可接受，但仍有优化空间。")
    else:
        notes.append("BER 较低，链路传输质量良好。")

    if q < 3:
        score -= 20; notes.append("Q 因子偏低，接收判决裕量不足。")
    elif q < 6:
        score -= 8; notes.append("Q 因子中等，适合做参数优化实验。")

    if osnr < 10:
        score -= 15; notes.append("OSNR 偏低，ASE 噪声或链路损耗影响明显。")
    if rx_power < -30:
        score -= 10; notes.append("接收功率较低，可能接近接收机灵敏度下限。")

    score = max(0, min(100, int(score)))
    if score >= 90: grade = "优"
    elif score >= 80: grade = "良"
    elif score >= 70: grade = "中"
    elif score >= 60: grade = "及格"
    else: grade = "不及格"
    return score, grade, notes

def generate_rule_based_report(params, metrics):
    score, grade, notes = evaluate_link_quality(metrics)
    notes_text = "\n".join([f"- {n}" for n in notes])
    if metrics["ber"] <= 1e-9:
        main_comment = "系统误码率较低，Q 因子和 OSNR 表现较好，当前参数组合适合作为可靠传输方案。"
    elif metrics["ber"] <= 1e-3:
        main_comment = "系统可以观察到明显的性能退化趋势，适合作为参数优化和链路预算训练案例。"
    else:
        main_comment = "系统误码率过高，当前链路参数不满足可靠通信要求，应重点调整发射功率、EDFA 增益、传输距离或调制格式。"

    return f"""# 光纤通信系统仿真实验报告

## 一、实验目的

通过虚拟仿真平台观察发射功率、传输距离、光纤衰减、色散、EDFA 增益和调制格式对接收光功率、OSNR、Q 因子与 BER 的影响，理解光纤通信系统链路预算和工程折中设计方法。

## 二、实验参数

- 工作波长：{params['wavelength']}
- 调制格式：{params['modulation']}
- 发射功率：{params['tx_power']} dBm
- 传输距离：{params['distance']} km
- 光纤衰减系数：{params['attenuation']} dB/km
- 色散系数：{params['dispersion']} ps/nm/km
- EDFA 增益：{params['edfa_gain']} dB

## 三、实验结果

- 接收光功率：{metrics['rx_power']:.2f} dBm
- OSNR：{metrics['osnr']:.2f} dB
- Q 因子：{metrics['q_factor']:.2f}
- BER：{metrics['ber']:.2e}
- 色散积：{metrics['dispersion_total']:.2f} ps/nm
- 非线性噪声估计：{metrics['nli_power_dbm']:.2f} dBm

## 四、实验数据分析

{main_comment}

系统诊断要点：

{notes_text}

## 五、改进建议

1. 当 BER 偏高时，可优先尝试降低传输距离、提高 OSNR 或选择鲁棒性更强的调制格式。
2. EDFA 增益不宜盲目增大，过高增益会引入 ASE 噪声并影响 OSNR。
3. 长距离传输时应关注色散积和脉冲展宽，必要时加入色散补偿或 DSP 均衡。
4. 高阶调制格式对 OSNR 更敏感，应结合目标 BER 和链路预算综合选择。

## 六、最终评分

- 规则库评分：{score} 分
- 实验等级：{grade}

## 七、课程思政评语

光纤通信是现代信息基础设施的重要支撑。实验过程中应坚持严谨求实的科学态度，理解工程系统中性能、成本、功耗和可靠性的综合折中，培养精益求精的工程素养。
"""

def show_report_dialog(report_content, title="🎓 智能实验报告"):
    @st.dialog(title)
    def _show_report():
        st.markdown(report_content)
        st.download_button("📥 下载报告 (Markdown)", report_content, "Lab_Report.md")
    _show_report()

@st.cache_data(ttl=10, show_spinner=False)
def fetch_local_model_ids(base_url, api_key):
    if not OPENAI_AVAILABLE:
        return [], "openai 依赖不可用"
    try:
        probe_client = OpenAI(base_url=base_url, api_key=api_key)
        response = probe_client.models.list()
        model_ids = [m.id for m in getattr(response, "data", []) if getattr(m, "id", None)]
        return model_ids, ""
    except Exception as exc:
        return [], str(exc)

def estimate_required_q(target_ber_exp):
    q_lookup = {
        3: 3.1, 4: 3.5, 5: 4.0, 6: 4.5, 7: 5.0,
        8: 5.5, 9: 6.0, 10: 6.4, 11: 6.8, 12: 7.2
    }
    return q_lookup.get(int(target_ber_exp), 6.0)

# --- 为核心物理计算添加缓存机制，避免重复计算 ---
@st.cache_data(show_spinner=False)
def calculate_simulation(params):
    """
    工程近似链路模型：链路预算 + EDFA ASE + 简化 GN 非线性噪声 + 色散功率代价 + 不同调制格式 BER 近似。
    该模型比原教学模型更接近物理实验趋势，但仍不替代 OptiSystem/VPI/SSFM 等高保真仿真。
    """
    h = 6.62607015e-34
    c = 2.99792458e8
    bit_rate = 25e9                 # 等效单通道速率，用于教学近似
    noise_bw = 12.5e9               # OSNR/电噪声等效带宽
    ref_bw = 12.5e9

    tx_power_dbm = float(params['tx_power'])
    distance_km = float(params['distance'])
    att_db = float(params['attenuation'])
    disp_psnmkm = float(params['dispersion'])
    edfa_gain_set_db = float(params['edfa_gain'])
    wavelength = params['wavelength']
    modulation = params['modulation']

    # 【新增】OptiSystem 对齐的发射端非理想参数。默认关闭，关闭时完全保持原模型。
    tx_impair_enabled = bool(params.get('tx_impair_enabled', False))
    tx_ext_ratio_db = float(params.get('tx_ext_ratio_db', 12.0))
    mzm_bias_v = float(params.get('mzm_bias_v', 2.0))
    mzm_vpi = max(float(params.get('mzm_vpi', 4.0)), 0.1)
    mzm_vpp = float(params.get('mzm_vpp', 4.0))
    laser_linewidth_mhz = float(params.get('laser_linewidth_mhz', 1.0))
    tx_rin_db_hz = float(params.get('tx_rin_db_hz', -150.0))
    tx_chirp_alpha = float(params.get('tx_chirp_alpha', 0.0))

    # 【新增】OptiSystem 对齐的光纤/EDFA/WDM/判决分析高级参数。默认关闭，关闭时保持原模型。
    fiber_advanced_enabled = bool(params.get('fiber_advanced_enabled', False))
    fiber_pmd_ps_sqrtkm = float(params.get('fiber_pmd_ps_sqrtkm', 0.05))
    fiber_gamma_wkm = float(params.get('fiber_gamma_wkm', 1.3))
    fiber_aeff_um2 = float(params.get('fiber_aeff_um2', 80.0))
    fiber_disp_slope = float(params.get('fiber_disp_slope', 0.08))
    link_connector_count = int(params.get('link_connector_count', 0))
    link_connector_loss_db = float(params.get('link_connector_loss_db', 0.25))
    splice_count = int(params.get('splice_count', 0))
    splice_loss_db = float(params.get('splice_loss_db', 0.05))
    span_length_setting_km = float(params.get('span_length_setting_km', 80.0))
    link_margin_db = float(params.get('link_margin_db', 3.0))

    edfa_advanced_enabled = bool(params.get('edfa_advanced_enabled', False))
    edfa_nf_db = float(params.get('edfa_nf_db', 5.5))
    edfa_psat_dbm = float(params.get('edfa_psat_dbm', 20.0))
    edfa_ase_bw_ghz = float(params.get('edfa_ase_bw_ghz', 12.5))
    edfa_gain_flatness_db = float(params.get('edfa_gain_flatness_db', 0.5))

    decision_eye_enabled = bool(params.get('decision_eye_enabled', False))
    rx_filter_bw_ghz = float(params.get('rx_filter_bw_ghz', 18.0))
    rx_filter_type = params.get('rx_filter_type', 'Bessel')
    decision_threshold = float(params.get('decision_threshold', 0.5))
    sampling_offset_ui = float(params.get('sampling_offset_ui', 0.0))
    clock_jitter_ps = float(params.get('clock_jitter_ps', 1.0))

    wdm_enabled = bool(params.get('wdm_enabled', False))
    wdm_channels = int(params.get('wdm_channels', 1))
    wdm_spacing_ghz = float(params.get('wdm_spacing_ghz', 100.0))
    mux_loss_db = float(params.get('mux_loss_db', 1.5))
    demux_loss_db = float(params.get('demux_loss_db', 1.5))
    wdm_xtalk_db = float(params.get('wdm_xtalk_db', -30.0))

    lam_m = 1550e-9 if "1550" in wavelength else 1310e-9
    nu = c / lam_m
    p_tx_w = 10 ** ((tx_power_dbm - 30) / 10.0)

    span_length_ref = span_length_setting_km if fiber_advanced_enabled else 80.0
    span_length_ref = max(10.0, span_length_ref)
    num_spans = max(1, int(np.ceil(distance_km / span_length_ref)))
    span_len_km = distance_km / num_spans
    span_loss_db = span_len_km * att_db
    total_loss_db = distance_km * att_db

    # EDFA 工程近似：增益不能无限增大，受饱和输出功率限制。
    nf_db = edfa_nf_db if edfa_advanced_enabled else 5.5
    nf_lin = 10 ** (nf_db / 10.0)
    psat_dbm = edfa_psat_dbm if edfa_advanced_enabled else 20.0
    noise_bw = max(edfa_ase_bw_ghz, 0.1) * 1e9 if edfa_advanced_enabled else noise_bw
    span_gain_db = min(edfa_gain_set_db, max(0.0, span_loss_db + 3.0))
    p_signal_after_span_dbm = tx_power_dbm
    p_ase_total_w = 0.0
    gain_saturation_penalty_db = 0.0

    for _ in range(num_spans):
        p_before_amp_dbm = p_signal_after_span_dbm - span_loss_db
        small_signal_out_dbm = p_before_amp_dbm + span_gain_db
        if small_signal_out_dbm > psat_dbm:
            gain_saturation_penalty_db += small_signal_out_dbm - psat_dbm
            actual_gain_db = max(0.0, psat_dbm - p_before_amp_dbm)
            p_signal_after_span_dbm = psat_dbm
        else:
            actual_gain_db = span_gain_db
            p_signal_after_span_dbm = small_signal_out_dbm
        gain_lin = 10 ** (actual_gain_db / 10.0)
        p_ase_total_w += 2.0 * h * nu * nf_lin * max(gain_lin - 1.0, 0.0) * noise_bw

    rx_power_dbm = p_signal_after_span_dbm

    # 光纤链路/WDM 额外无源损耗：连接器、熔接、MUX/DEMUX 与工程链路余量。
    link_connector_total_loss_db = 0.0
    splice_total_loss_db = 0.0
    passive_wdm_loss_db = 0.0
    link_margin_penalty_db = 0.0
    if fiber_advanced_enabled:
        link_connector_total_loss_db = max(link_connector_count, 0) * max(link_connector_loss_db, 0.0)
        splice_total_loss_db = max(splice_count, 0) * max(splice_loss_db, 0.0)
        link_margin_penalty_db = max(link_margin_db, 0.0)
    if wdm_enabled:
        passive_wdm_loss_db = max(mux_loss_db, 0.0) + max(demux_loss_db, 0.0)
    rx_power_dbm -= (link_connector_total_loss_db + splice_total_loss_db + passive_wdm_loss_db + link_margin_penalty_db)
    p_rx_w = 10 ** ((rx_power_dbm - 30) / 10.0)

    # 接收机热噪声/暗电流等效噪声底：用于让接收灵敏度影响 BER 趋势。
    rx_sensitivity_dbm = {'OOK': -28.0, 'FSK': -27.0, 'DPSK': -30.0, 'PAM4': -20.0,
                          'QPSK': -24.0, '8-PSK': -20.0, '16-QAM': -17.0}.get(modulation, -22.0)
    p_sens_w = 10 ** ((rx_sensitivity_dbm - 30) / 10.0)
    p_rx_elec_noise_w = p_sens_w / 25.0
    # 保存原始接收机模型，用于后面生成“加入接收机噪声前/后”的对比结果。
    base_rx_sensitivity_dbm = rx_sensitivity_dbm
    base_p_rx_elec_noise_w = p_rx_elec_noise_w

    # 【新增】接收机偏置电压与噪声模块：默认关闭，关闭时保持原有模型输出不变。
    receiver_noise_enabled = bool(params.get('receiver_noise_enabled', False))
    rx_bias_voltage = float(params.get('rx_bias_voltage', 5.0))
    rx_bandwidth_ghz = float(params.get('rx_bandwidth_ghz', 12.5))
    rx_responsivity = float(params.get('rx_responsivity', 0.85))
    rx_load_ohm = float(params.get('rx_load_ohm', 50.0))
    rx_dark_current_na = float(params.get('rx_dark_current_na', 2.0))
    rx_noise_scale = float(params.get('rx_noise_scale', 1.0))
    rx_bias_penalty_db = 0.0
    rx_noise_penalty_db = 0.0
    p_thermal_noise_eq_w = 0.0
    p_shot_noise_eq_w = 0.0
    dark_current_effective_na = rx_dark_current_na
    effective_responsivity = rx_responsivity

    if receiver_noise_enabled:
        q_e = 1.602176634e-19
        k_b = 1.380649e-23
        temp_k = 300.0
        b_rx = max(rx_bandwidth_ghz, 0.1) * 1e9
        r_load = max(rx_load_ohm, 1.0)
        # 偏置不足时耗尽层不充分，等效响应度下降；偏置过高时暗电流上升。
        bias_collection_factor = 0.55 + 0.45 * (1.0 - np.exp(-max(rx_bias_voltage, 0.0) / 3.0))
        effective_responsivity = max(rx_responsivity * bias_collection_factor, 1e-6)
        if rx_bias_voltage < 2.0:
            rx_bias_penalty_db = min(8.0, (2.0 - rx_bias_voltage) * 2.0)
        elif rx_bias_voltage > 12.0:
            rx_bias_penalty_db = min(6.0, (rx_bias_voltage - 12.0) * 0.8)
        dark_current_effective_na = rx_dark_current_na * (1.0 + max(rx_bias_voltage - 8.0, 0.0) ** 2 / 16.0)
        i_photo = effective_responsivity * p_rx_w
        i_dark = dark_current_effective_na * 1e-9
        i_thermal_var = 4.0 * k_b * temp_k * b_rx / r_load
        i_shot_var = 2.0 * q_e * max(i_photo + i_dark, 0.0) * b_rx
        p_thermal_noise_eq_w = (math.sqrt(max(i_thermal_var, 0.0)) / effective_responsivity) ** 2
        p_shot_noise_eq_w = (math.sqrt(max(i_shot_var, 0.0)) / effective_responsivity) ** 2
        physical_rx_noise_w = (p_thermal_noise_eq_w + p_shot_noise_eq_w) * max(rx_noise_scale, 0.0)
        p_rx_elec_noise_w = max(p_rx_elec_noise_w, physical_rx_noise_w)

        # 教学可视化代价：真实接收机热噪声/散粒噪声的绝对功率有时远低于当前链路噪声，
        # 会导致滑块变化不明显。这里把偏置不足、带宽变宽、暗电流升高、响应度下降
        # 折算为等效 SNR 代价，使学生能直接在 Q 因子和 BER 中看到接收机非理想影响。
        responsivity_penalty_db = max(0.0, (0.85 - effective_responsivity) / 0.85 * 6.0)
        bandwidth_penalty_db = max(0.0, (rx_bandwidth_ghz - 12.5) / 37.5 * 8.0)
        dark_penalty_db = min(8.0, np.log10(1.0 + dark_current_effective_na / 5.0) * 4.0)
        load_penalty_db = max(0.0, np.log10(50.0 / max(r_load, 1.0)) * 2.0)
        scale_penalty_db = max(0.0, rx_noise_scale - 1.0) * 2.5
        rx_noise_penalty_db = min(22.0, (responsivity_penalty_db + bandwidth_penalty_db + dark_penalty_db + load_penalty_db + scale_penalty_db) * max(rx_noise_scale, 0.0))
        rx_sensitivity_dbm += rx_bias_penalty_db + rx_noise_penalty_db

    # 简化 GN 非线性噪声：用于呈现高入纤功率时代价迅速上升的趋势。
    gamma = (fiber_gamma_wkm * 1e-3) if fiber_advanced_enabled else 1.3e-3
    alpha_np_per_m = att_db / (10 * np.log10(np.e)) / 1e3
    span_len_m = span_len_km * 1e3
    leff = (1 - np.exp(-alpha_np_per_m * span_len_m)) / max(alpha_np_per_m, 1e-12)
    D = max(abs(disp_psnmkm), 0.1)
    beta2 = abs(D * 1e-6 * (lam_m ** 2) / (2 * np.pi * c))
    aeff_factor = (80.0 / max(fiber_aeff_um2, 20.0)) ** 2 if fiber_advanced_enabled else 1.0
    wdm_nli_factor = (1.0 + 0.08 * max(wdm_channels - 1, 0)) if wdm_enabled else 1.0
    eta_nli = 1.5e-3 * (gamma * leff) ** 2 / max(beta2 / 1e-26, 0.3) * aeff_factor * wdm_nli_factor
    p_nli_w = max(1e-24, eta_nli * (p_tx_w ** 3) * num_spans)

    total_noise_w = p_ase_total_w + p_nli_w + p_rx_elec_noise_w + 1e-24
    snr_linear = max(p_rx_w / total_noise_w, 1e-12)
    snr_db_raw = 10 * np.log10(snr_linear)

    # 原始接收机模型的 SNR，用于对比显示。
    base_total_noise_w = p_ase_total_w + p_nli_w + base_p_rx_elec_noise_w + 1e-24
    base_snr_linear = max(p_rx_w / base_total_noise_w, 1e-12)
    base_snr_db_raw = 10 * np.log10(base_snr_linear)

    # 色散功率代价：按调制格式差异给出更真实的趋势。
    total_disp = distance_km * disp_psnmkm
    disp_tolerance = {'OOK': 1200, 'FSK': 1000, 'DPSK': 1800, 'PAM4': 700,
                      'QPSK': 2200, '8-PSK': 1400, '16-QAM': 900}.get(modulation, 1000)
    dispersion_penalty_db = 0.0
    if abs(total_disp) > disp_tolerance:
        dispersion_penalty_db = min(12.0, 3.0 * ((abs(total_disp) / disp_tolerance) - 1.0) ** 2)

    fiber_pmd_penalty_db = 0.0
    fiber_slope_penalty_db = 0.0
    fiber_nli_penalty_db = 0.0
    pmd_dgd_ps = 0.0
    if fiber_advanced_enabled:
        pmd_dgd_ps = max(fiber_pmd_ps_sqrtkm, 0.0) * math.sqrt(max(distance_km, 0.0))
        symbol_period_ps = 1000.0 / 25.0  # 25 Gbaud 教学近似，单位 ps
        pmd_ratio = pmd_dgd_ps / max(symbol_period_ps, 1e-9)
        mod_pmd_weight = {'OOK': 1.0, 'PAM4': 1.45, 'QPSK': 1.2, '8-PSK': 1.45, '16-QAM': 1.7}.get(modulation, 1.0)
        fiber_pmd_penalty_db = min(8.0, (pmd_ratio / 0.12) ** 2 * 1.2 * mod_pmd_weight)
        fiber_slope_penalty_db = min(4.0, abs(fiber_disp_slope) * distance_km / 240.0 * (1.0 if wdm_enabled else 0.35))
        gamma_ref = 1.3
        fiber_nli_penalty_db = min(8.0, max(0.0, (fiber_gamma_wkm / gamma_ref - 1.0)) * max(tx_power_dbm + 3.0, 0.0) / 8.0 * 3.0 * (80.0 / max(fiber_aeff_um2, 20.0)))

    edfa_flatness_penalty_db = min(5.0, max(edfa_gain_flatness_db, 0.0) * 0.8) if edfa_advanced_enabled else 0.0

    wdm_xtalk_penalty_db = 0.0
    wdm_spacing_penalty_db = 0.0
    if wdm_enabled:
        xtalk_lin = 10 ** (wdm_xtalk_db / 10.0)
        wdm_xtalk_penalty_db = min(9.0, 10 * np.log10(1.0 + max(wdm_channels - 1, 0) * xtalk_lin * 80.0))
        wdm_spacing_penalty_db = min(6.0, max(0.0, (100.0 - wdm_spacing_ghz) / 50.0) * max(wdm_channels - 1, 0) * 0.45)

    # 【新增】发射端非理想代价：消光比、MZM 偏置/驱动、激光器 RIN、线宽、啁啾。
    tx_er_penalty_db = 0.0
    tx_mzm_bias_penalty_db = 0.0
    tx_mzm_drive_penalty_db = 0.0
    tx_linewidth_penalty_db = 0.0
    tx_rin_penalty_db = 0.0
    tx_chirp_penalty_db = 0.0
    tx_total_penalty_db = 0.0
    tx_modulation_depth = 1.0
    tx_bias_offset_ratio = 0.0
    tx_mod_weight = {'OOK': 1.0, 'FSK': 0.9, 'DPSK': 1.1, 'PAM4': 1.45,
                     'QPSK': 1.25, '8-PSK': 1.55, '16-QAM': 1.75}.get(modulation, 1.0)
    if tx_impair_enabled:
        # 消光比越低，“0”码泄漏越大，眼图垂直开口减小。
        tx_er_penalty_db = min(10.0, max(0.0, (10.0 - tx_ext_ratio_db) * 0.75 * tx_mod_weight))

        # MZM 最佳偏置近似为 Vpi/2，即 quadrature 点；偏离越大，调制线性和眼图越差。
        optimum_bias_v = 0.5 * mzm_vpi
        tx_bias_offset_ratio = abs(mzm_bias_v - optimum_bias_v) / mzm_vpi
        tx_mzm_bias_penalty_db = min(10.0, (tx_bias_offset_ratio / 0.25) ** 2 * 2.0 * tx_mod_weight)

        # Vpp 太小：调制深度不足；Vpp 过大：过驱动带来非线性失真。
        tx_modulation_depth = float(np.clip(abs(np.sin(np.pi * mzm_vpp / (2.0 * mzm_vpi))), 0.02, 1.0))
        underdrive_penalty = max(0.0, (0.75 - tx_modulation_depth) / 0.75) * 7.0
        overdrive_penalty = max(0.0, (mzm_vpp / mzm_vpi - 1.35)) * 4.0
        tx_mzm_drive_penalty_db = min(9.0, (underdrive_penalty + overdrive_penalty) * tx_mod_weight)

        # 线宽主要影响相干调制星座图；OOK/PAM4 中影响较弱。
        linewidth_weight = {'OOK': 0.15, 'FSK': 0.3, 'PAM4': 0.25, 'DPSK': 0.9,
                            'QPSK': 1.0, '8-PSK': 1.3, '16-QAM': 1.6}.get(modulation, 0.8)
        tx_linewidth_penalty_db = min(8.0, np.log10(1.0 + max(laser_linewidth_mhz, 0.0)) * 2.2 * linewidth_weight)

        # RIN 越接近 -130 dB/Hz 越差；低于 -155 dB/Hz 基本可认为影响很小。
        rin_weight = {'OOK': 1.2, 'PAM4': 1.5, 'FSK': 0.9, 'DPSK': 0.8,
                      'QPSK': 0.7, '8-PSK': 0.8, '16-QAM': 1.0}.get(modulation, 1.0)
        tx_rin_penalty_db = min(8.0, max(0.0, (tx_rin_db_hz + 155.0) / 10.0) * rin_weight)

        # 直接调制 LD 的啁啾与色散耦合，长距离/大色散时惩罚更明显。
        chirp_weight = {'OOK': 1.0, 'PAM4': 1.4, 'FSK': 0.8, 'DPSK': 1.0,
                        'QPSK': 0.9, '8-PSK': 1.1, '16-QAM': 1.3}.get(modulation, 1.0)
        tx_chirp_penalty_db = min(10.0, abs(tx_chirp_alpha) * abs(total_disp) / 1100.0 * chirp_weight)

        tx_total_penalty_db = min(28.0, tx_er_penalty_db + tx_mzm_bias_penalty_db +
                                  tx_mzm_drive_penalty_db + tx_linewidth_penalty_db +
                                  tx_rin_penalty_db + tx_chirp_penalty_db)

    # 其它工程惩罚项：连接器/滤波器/DSP非理想项，避免理想公式过于乐观。
    implementation_penalty_db = {'OOK': 2.0, 'FSK': 3.0, 'DPSK': 2.5, 'PAM4': 4.5,
                                 'QPSK': 3.0, '8-PSK': 4.0, '16-QAM': 5.0}.get(modulation, 3.0)
    link_advanced_penalty_db = (fiber_pmd_penalty_db + fiber_slope_penalty_db + fiber_nli_penalty_db +
                                edfa_flatness_penalty_db + wdm_xtalk_penalty_db + wdm_spacing_penalty_db)
    decision_filter_penalty_db = 0.0
    decision_threshold_penalty_db = 0.0
    sampling_jitter_penalty_db = 0.0
    if decision_eye_enabled:
        # 低通滤波器带宽太窄会造成 ISI，太宽会引入更多噪声；判决门限和采样偏移直接压缩眼图开口。
        filter_weight = {'Bessel': 0.9, 'Gaussian': 1.0, 'Butterworth': 1.15}.get(rx_filter_type, 1.0)
        opt_bw = 18.0 if modulation in ['OOK', 'DPSK', 'FSK'] else 25.0
        narrow_penalty = max(0.0, (opt_bw - rx_filter_bw_ghz) / opt_bw) * 7.0
        wide_penalty = max(0.0, (rx_filter_bw_ghz - opt_bw * 1.6) / (opt_bw * 1.6)) * 4.0
        decision_filter_penalty_db = min(9.0, (narrow_penalty + wide_penalty) * filter_weight)
        decision_threshold_penalty_db = min(8.0, abs(decision_threshold - 0.5) / 0.25 * 4.0)
        sampling_jitter_penalty_db = min(10.0, (abs(sampling_offset_ui) / 0.25) ** 2 * 5.0 + max(clock_jitter_ps - 2.0, 0.0) / 10.0 * 3.0)

    analysis_penalty_db = decision_filter_penalty_db + decision_threshold_penalty_db + sampling_jitter_penalty_db
    base_effective_snr_db = base_snr_db_raw - dispersion_penalty_db - implementation_penalty_db - gain_saturation_penalty_db - link_advanced_penalty_db
    base_effective_snr_linear = max(10 ** (base_effective_snr_db / 10.0), 1e-12)
    effective_snr_db = snr_db_raw - dispersion_penalty_db - implementation_penalty_db - gain_saturation_penalty_db - link_advanced_penalty_db - analysis_penalty_db - tx_total_penalty_db - rx_bias_penalty_db - rx_noise_penalty_db
    effective_snr_linear = max(10 ** (effective_snr_db / 10.0), 1e-12)

    def qfunc(x):
        return 0.5 * math.erfc(float(x) / math.sqrt(2))

    def ber_q_from_snr(snr_lin, modulation_name):
        if modulation_name == 'OOK':
            q_val = math.sqrt(snr_lin)
            ber_val = qfunc(q_val)
        elif modulation_name == 'DPSK':
            q_val = math.sqrt(2 * snr_lin)
            ber_val = 0.5 * math.exp(-snr_lin)
        elif modulation_name == 'FSK':
            q_val = math.sqrt(snr_lin)
            ber_val = 0.5 * math.exp(-0.5 * snr_lin)
        elif modulation_name == 'PAM4':
            q_val = math.sqrt(snr_lin) / 3.0
            ber_val = 0.75 * qfunc(math.sqrt(0.8 * snr_lin))
        elif modulation_name == 'QPSK':
            q_val = math.sqrt(2 * snr_lin)
            ber_val = qfunc(math.sqrt(2 * snr_lin))
        elif modulation_name == '8-PSK':
            q_val = math.sqrt(2 * snr_lin) * math.sin(math.pi / 8)
            ber_val = (2 / 3) * qfunc(math.sqrt(2 * snr_lin) * math.sin(math.pi / 8))
        elif modulation_name == '16-QAM':
            q_val = math.sqrt(snr_lin / 5.0)
            ber_val = (3 / 8) * math.erfc(math.sqrt(snr_lin / 10.0))
        else:
            q_val = math.sqrt(snr_lin)
            ber_val = qfunc(q_val)
        return max(0.0, q_val), min(1.0, max(1e-18, ber_val))

    base_q_factor, base_ber = ber_q_from_snr(base_effective_snr_linear, modulation)
    q_factor, ber = ber_q_from_snr(effective_snr_linear, modulation)
    osnr_db = 10 * np.log10(max(p_rx_w / (p_ase_total_w + p_nli_w + 1e-24), 1e-12))
    osnr_db_01nm = osnr_db + 10 * np.log10(noise_bw / ref_bw)

    return {
        "rx_power": rx_power_dbm,
        "osnr": osnr_db_01nm,
        "q_factor": max(0.0, q_factor),
        "ber": ber,
        "dispersion_total": total_disp,
        "nli_power_dbm": 10*np.log10(p_nli_w*1e3) if p_nli_w > 0 else -100,
        "ase_power_dbm": 10*np.log10(p_ase_total_w*1e3) if p_ase_total_w > 0 else -100,
        "rx_sensitivity_dbm": rx_sensitivity_dbm,
        "sensitivity_margin_db": rx_power_dbm - rx_sensitivity_dbm,
        "dispersion_penalty_db": dispersion_penalty_db,
        "gain_saturation_penalty_db": gain_saturation_penalty_db,
        "base_q_factor": max(0.0, base_q_factor),
        "base_ber": base_ber,
        "base_rx_sensitivity_dbm": base_rx_sensitivity_dbm,
        "effective_snr_db": effective_snr_db,
        "base_effective_snr_db": base_effective_snr_db,
        "tx_impair_enabled": tx_impair_enabled,
        "tx_total_penalty_db": tx_total_penalty_db,
        "tx_er_penalty_db": tx_er_penalty_db,
        "tx_mzm_bias_penalty_db": tx_mzm_bias_penalty_db,
        "tx_mzm_drive_penalty_db": tx_mzm_drive_penalty_db,
        "tx_linewidth_penalty_db": tx_linewidth_penalty_db,
        "tx_rin_penalty_db": tx_rin_penalty_db,
        "tx_chirp_penalty_db": tx_chirp_penalty_db,
        "tx_modulation_depth": tx_modulation_depth,
        "tx_bias_offset_ratio": tx_bias_offset_ratio,
        "tx_ext_ratio_db": tx_ext_ratio_db,
        "mzm_bias_v": mzm_bias_v,
        "mzm_vpi": mzm_vpi,
        "mzm_vpp": mzm_vpp,
        "laser_linewidth_mhz": laser_linewidth_mhz,
        "tx_rin_db_hz": tx_rin_db_hz,
        "tx_chirp_alpha": tx_chirp_alpha,
        "rx_bias_voltage": rx_bias_voltage,
        "rx_bias_penalty_db": rx_bias_penalty_db,
        "rx_noise_penalty_db": rx_noise_penalty_db,
        "rx_total_penalty_db": rx_bias_penalty_db + rx_noise_penalty_db,
        "rx_effective_responsivity": effective_responsivity,
        "rx_dark_current_na": dark_current_effective_na,
        "rx_thermal_noise_dbm": 10*np.log10(p_thermal_noise_eq_w*1e3) if p_thermal_noise_eq_w > 0 else -100,
        "rx_shot_noise_dbm": 10*np.log10(p_shot_noise_eq_w*1e3) if p_shot_noise_eq_w > 0 else -100,
        "receiver_noise_enabled": receiver_noise_enabled,
        "fiber_advanced_enabled": fiber_advanced_enabled,
        "fiber_pmd_ps_sqrtkm": fiber_pmd_ps_sqrtkm,
        "fiber_gamma_wkm": fiber_gamma_wkm,
        "fiber_aeff_um2": fiber_aeff_um2,
        "fiber_disp_slope": fiber_disp_slope,
        "pmd_dgd_ps": pmd_dgd_ps,
        "fiber_pmd_penalty_db": fiber_pmd_penalty_db,
        "fiber_slope_penalty_db": fiber_slope_penalty_db,
        "fiber_nli_penalty_db": fiber_nli_penalty_db,
        "link_connector_total_loss_db": link_connector_total_loss_db,
        "splice_total_loss_db": splice_total_loss_db,
        "link_margin_penalty_db": link_margin_penalty_db,
        "link_advanced_penalty_db": link_advanced_penalty_db,
        "edfa_advanced_enabled": edfa_advanced_enabled,
        "edfa_nf_db": nf_db,
        "edfa_psat_dbm": psat_dbm,
        "edfa_ase_bw_ghz": edfa_ase_bw_ghz,
        "edfa_flatness_penalty_db": edfa_flatness_penalty_db,
        "decision_eye_enabled": decision_eye_enabled,
        "rx_filter_bw_ghz": rx_filter_bw_ghz,
        "rx_filter_type": rx_filter_type,
        "decision_threshold": decision_threshold,
        "sampling_offset_ui": sampling_offset_ui,
        "clock_jitter_ps": clock_jitter_ps,
        "decision_filter_penalty_db": decision_filter_penalty_db,
        "decision_threshold_penalty_db": decision_threshold_penalty_db,
        "sampling_jitter_penalty_db": sampling_jitter_penalty_db,
        "analysis_penalty_db": analysis_penalty_db,
        "wdm_enabled": wdm_enabled,
        "wdm_channels": wdm_channels,
        "wdm_spacing_ghz": wdm_spacing_ghz,
        "mux_demux_loss_db": passive_wdm_loss_db,
        "wdm_xtalk_penalty_db": wdm_xtalk_penalty_db,
        "wdm_spacing_penalty_db": wdm_spacing_penalty_db,
        "num_spans": num_spans,
        "span_loss_db": span_loss_db,
        "model_note": "工程近似：链路预算 + EDFA ASE + GN非线性 + 色散代价 + 调制格式BER近似"
    }

@st.cache_data(show_spinner=False)
def run_coherent_dsp_demo(modulation, q_factor, dispersion, phase_noise_std=0.05):
    np.random.seed(42) # 固定种子避免无意义的重绘抖动
    n_symbols = 800  
    
    if 'QAM' in modulation or 'PSK' in modulation or modulation == 'QPSK':
        if modulation == '16-QAM':
            levels = [-3, -1, 1, 3]
            base_points = [x + 1j*y for x in levels for y in levels]
            base_points = np.array(base_points) / np.sqrt(np.mean(np.abs(np.array(base_points))**2)) * np.sqrt(2)
        else:
            base_points = [1+1j, 1-1j, -1+1j, -1-1j]
    else:
        base_points = [1+0j, -1+0j]

    tx_sig = np.random.choice(base_points, n_symbols)
    
    noise_scale = 1.0 / (q_factor + 1e-9) * 0.7
    ase_noise = np.random.normal(0, noise_scale, n_symbols) + 1j * np.random.normal(0, noise_scale, n_symbols)
    
    pn_sequence = np.cumsum(np.random.normal(0, phase_noise_std, n_symbols))
    phase_rotation = np.exp(1j * pn_sequence)
    
    cd_severity = dispersion * 0.003
    cd_effect = np.exp(1j * np.random.randn(n_symbols) * cd_severity * 5)
    
    sig_stage_3 = tx_sig + ase_noise
    sig_stage_2 = sig_stage_3 * phase_rotation
    sig_stage_1 = sig_stage_2 * cd_effect
    
    return sig_stage_1, sig_stage_2, sig_stage_3


def generate_eye_diagram_data(metrics, params, n_bits=160, samples_per_bit=32):
    """生成用于教学显示的 NRZ/PAM4 眼图。它把 Q/BER、发射端、光纤、接收机、判决器代价转化为眼图开口变化。"""
    rng = np.random.default_rng(2026)
    modulation = params.get("modulation", "OOK")
    if modulation == "PAM4":
        symbols = rng.integers(0, 4, n_bits)
        levels = np.array([-3, -1, 1, 3], dtype=float) / 3.0
        data = levels[symbols]
    else:
        symbols = rng.integers(0, 2, n_bits)
        data = 2 * symbols.astype(float) - 1

    sps = samples_per_bit
    waveform = np.repeat(data, sps)
    t = np.arange(len(waveform)) / sps

    # 带宽不足和色散/PMD 会让边沿变缓，相当于低通卷积。
    filter_pen = metrics.get("decision_filter_penalty_db", 0.0)
    disp_pen = metrics.get("dispersion_penalty_db", 0.0) + metrics.get("fiber_pmd_penalty_db", 0.0)
    smooth_len = int(np.clip(2 + filter_pen * 1.4 + disp_pen * 0.9, 2, sps * 2))
    kernel = np.hanning(max(3, smooth_len))
    kernel = kernel / max(kernel.sum(), 1e-9)
    waveform = np.convolve(waveform, kernel, mode="same")

    # 根据有效 SNR、RIN、接收机噪声和串扰注入幅度噪声。
    q = max(metrics.get("q_factor", 1.0), 0.15)
    noise_std = np.clip(0.18 / q + metrics.get("tx_rin_penalty_db", 0.0) * 0.015 + metrics.get("wdm_xtalk_penalty_db", 0.0) * 0.012, 0.01, 0.42)
    waveform = waveform + rng.normal(0, noise_std, len(waveform))

    # 采样偏移和时钟抖动压缩眼宽。
    ui_shift = metrics.get("sampling_offset_ui", 0.0)
    jitter_ps = metrics.get("clock_jitter_ps", 0.0)
    jitter_ui = np.clip(jitter_ps / 40.0, 0.0, 0.35)

    segments = []
    seg_t = np.linspace(-1, 1, 2 * sps, endpoint=False)
    for i in range(2, n_bits - 2):
        start = i * sps + int(ui_shift * sps) + int(rng.normal(0, jitter_ui * sps))
        end = start + 2 * sps
        if start >= 0 and end <= len(waveform):
            segments.append(waveform[start:end])
    if not segments:
        segments = [waveform[:2*sps]]
    arr = np.array(segments)

    center = arr[:, sps]
    if modulation == "PAM4":
        eye_height = max(0.0, 2.0 / 3.0 - 2.0 * np.std(center))
    else:
        ones = center[center > 0]
        zeros = center[center <= 0]
        if len(ones) and len(zeros):
            eye_height = max(0.0, np.mean(ones) - np.mean(zeros) - 3 * (np.std(ones) + np.std(zeros)))
        else:
            eye_height = max(0.0, 2.0 - 6 * np.std(center))
    eye_width = max(0.0, 1.0 - abs(ui_shift) * 1.7 - jitter_ui * 1.8 - filter_pen / 14.0 - disp_pen / 18.0)
    noise_margin = max(0.0, eye_height / 2.0 - abs(metrics.get("decision_threshold", 0.5) - 0.5))
    return seg_t, arr, {"eye_height": eye_height, "eye_width": eye_width, "noise_margin": noise_margin, "jitter_ui": jitter_ui, "noise_std": noise_std}

def encode_cmi(bits):
    encoded_bits = []
    last_one_level = 1 
    for b in bits:
        if b == 0: encoded_bits.extend([0, 1])
        else:
            if last_one_level == 1:
                encoded_bits.extend([0, 0]); last_one_level = 0
            else:
                encoded_bits.extend([1, 1]); last_one_level = 1
    return encoded_bits

def encode_5b6b(bits):
    """工程化 5B6B 教学编码：按运行数字选择 6 bit 码字，尽量保持直流平衡并避免长连 0/1。"""
    def to_dec(arr):
        return int("".join(str(x) for x in arr), 2)

    def max_run(seq):
        best = cur = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    # 生成候选码字：优先使用 3/3 平衡或 4/2、2/4 弱不平衡码字，过滤长连码。
    candidates = []
    for val in range(64):
        seq = [(val >> b) & 1 for b in range(5, -1, -1)]
        ones = sum(seq)
        disparity = ones - (6 - ones)
        if abs(disparity) <= 2 and max_run(seq) <= 4:
            candidates.append((seq, disparity, max_run(seq), val))
    candidates.sort(key=lambda item: (abs(item[1]), item[2], item[3]))

    # 为 32 个 5bit 输入分配互不重复的正/负候选码字。
    codebook = {}
    used = set()
    for data_val in range(32):
        desired = (bin(data_val).count('1') - (5 - bin(data_val).count('1')))
        pool = [c for c in candidates if c[3] not in used]
        if not pool:
            break
        chosen = min(pool, key=lambda c: (abs(c[1] + desired * 0.2), c[2], abs(c[3] - data_val * 2)))
        used.add(chosen[3])
        seq = chosen[0]
        inv = [1 - x for x in seq]
        codebook[data_val] = (seq, inv)

    encoded_bits = []
    running_disparity = 0
    padding = len(bits) % 5
    work_bits = bits[:]
    if padding != 0:
        work_bits.extend([0] * (5 - padding))

    for i in range(0, len(work_bits), 5):
        val = to_dec(work_bits[i:i+5])
        pos_code, neg_code = codebook.get(val, (work_bits[i:i+5] + [0], work_bits[i:i+5] + [1]))
        disp_pos = sum(pos_code) - (6 - sum(pos_code))
        disp_neg = sum(neg_code) - (6 - sum(neg_code))
        if abs(running_disparity + disp_pos) <= abs(running_disparity + disp_neg):
            out_chunk = pos_code
            running_disparity += disp_pos
        else:
            out_chunk = neg_code
            running_disparity += disp_neg
        encoded_bits.extend(out_chunk)
    return encoded_bits

def apply_channel_effects(digital_signal, osnr, bandwidth_factor=0.2):
    samples_per_bit = 20
    analog_signal = []
    for bit in digital_signal:
        analog_signal.extend([bit] * samples_per_bit)
    analog_signal = np.array(analog_signal, dtype=float)
    window_size = int(samples_per_bit * bandwidth_factor * 5) 
    if window_size < 1: window_size = 1
    kernel = np.ones(window_size) / window_size
    filtered_signal = np.convolve(analog_signal, kernel, mode='same')
    try:
        snr_linear = 10 ** (osnr / 10.0)
        noise_level = 1.0 / np.sqrt(snr_linear + 1e-9) * 0.8
    except:
        noise_level = 0.1
    noise = np.random.normal(0, noise_level, len(filtered_signal))
    received_signal = filtered_signal + noise
    t = np.linspace(0, len(digital_signal), len(received_signal))
    return t, received_signal, analog_signal

def simulate_otdr_trace(length_km, events_config, attenuation_coeff=0.20, pulse_width_ns=50.0,
                        dynamic_range_db=38.0, avg_count=16, group_index=1.468):
    """生成更接近真实 OTDR 的后向散射曲线：包含脉冲宽度、动态范围、事件死区、瑞利散射噪声和菲涅耳反射。"""
    num_points = 2400
    x_dist = np.linspace(0, length_km * 1.08, num_points)
    dx_km = x_dist[1] - x_dist[0]

    # OTDR 空间分辨率约 c*tau/(2n)，折合 km。
    resolution_km = max(0.001, (3e8 * pulse_width_ns * 1e-9 / (2 * group_index)) / 1000)
    sigma_pts = max(1.0, resolution_km / max(dx_km, 1e-6) / 2.355)
    kernel_half = int(max(4, sigma_pts * 5))
    kx = np.arange(-kernel_half, kernel_half + 1)
    pulse_kernel = np.exp(-(kx ** 2) / (2 * sigma_pts ** 2))
    pulse_kernel = pulse_kernel / pulse_kernel.max()

    # 双程损耗使背向散射斜率约为 2*alpha；为了视觉上与仪器读数接近，保留 dB 迹线。
    y_power = -2.0 * attenuation_coeff * x_dist
    noise_floor = -dynamic_range_db
    ground_truth = []

    def add_reflection(idx, peak_db):
        lo = max(0, idx - kernel_half)
        hi = min(len(y_power), idx + kernel_half + 1)
        klo = kernel_half - (idx - lo)
        khi = kernel_half + (hi - idx)
        y_power[lo:hi] = np.maximum(y_power[lo:hi], y_power[idx] + peak_db * pulse_kernel[klo:khi])

    sorted_events = sorted(events_config, key=lambda e: e['pos'])
    for event in sorted_events:
        pos = float(event['pos'])
        e_type = event['type']
        idx = int((np.abs(x_dist - pos)).argmin())

        if e_type == 'connector':
            event_loss, refl = 0.45, 7.0
            add_reflection(idx, refl)
            y_power[idx+1:] -= event_loss
            ground_truth.append(f"位置 {pos:.1f}km: 活动连接器，插入损耗约 {event_loss:.2f} dB，存在菲涅耳反射峰")
        elif e_type == 'fusion':
            event_loss, refl = 0.08, 0.5
            add_reflection(idx, refl)
            y_power[idx+1:] -= event_loss
            ground_truth.append(f"位置 {pos:.1f}km: 熔接点，非反射微损耗约 {event_loss:.2f} dB")
        elif e_type == 'bend':
            event_loss = 2.5
            y_power[idx+1:] -= event_loss
            ground_truth.append(f"位置 {pos:.1f}km: 宏弯/挤压故障，非反射损耗台阶约 {event_loss:.1f} dB")
        elif e_type == 'break':
            add_reflection(idx, 10.0)
            dead_pts = int(max(2, resolution_km / max(dx_km, 1e-6)))
            y_power[idx+dead_pts:] = noise_floor
            ground_truth.append(f"位置 {pos:.1f}km: 光纤断裂，强反射后进入噪声底")

    # 噪声与平均次数相关：平均越多，噪声越低；越接近噪声底波动越明显。
    rng = np.random.default_rng(42)
    noise_std = max(0.03, 0.55 / np.sqrt(max(avg_count, 1)))
    distance_noise = 0.015 * (x_dist / max(length_km, 1)) ** 2 * dynamic_range_db
    y_power += rng.normal(0, noise_std + distance_noise, num_points)
    y_power = np.maximum(y_power, noise_floor + rng.normal(0, 0.35, num_points))

    return x_dist, y_power, ground_truth

def ai_analyze_otdr(trace_y, x_axis, pulse_width_ns=50.0, dynamic_range_db=38.0):
    """基于平滑导数与死区跳过的 OTDR 事件检测，比简单差分阈值更稳定。"""
    if len(trace_y) < 5:
        return pd.DataFrame([])
    y = np.array(trace_y, dtype=float)
    x = np.array(x_axis, dtype=float)
    dx = max(x[1] - x[0], 1e-6)
    group_index = 1.468
    resolution_km = max(0.001, (3e8 * pulse_width_ns * 1e-9 / (2 * group_index)) / 1000)
    dead_zone_pts = int(max(8, resolution_km / dx * 3))

    kernel = np.ones(9) / 9
    y_smooth = np.convolve(y, kernel, mode='same')
    diff = np.diff(y_smooth)
    peak_thresh = 2.2
    drop_thresh = -0.45
    noise_floor = -dynamic_range_db

    events_found = []
    i = 5
    while i < len(diff) - 5:
        current_pos = x[i]
        if y_smooth[i] <= noise_floor + 2.0:
            i += 1
            continue
        local_peak = y_smooth[i] - np.median(y_smooth[max(0, i-20):i+1])
        local_drop = y_smooth[min(len(y_smooth)-1, i+dead_zone_pts)] - y_smooth[i]

        if local_peak > peak_thresh:
            event_type = "强反射事件（连接器/端面/断裂）"
            if y_smooth[min(len(y_smooth)-1, i + dead_zone_pts)] <= noise_floor + 4:
                event_type = "光纤断裂/末端事件"
            events_found.append({
                "位置 (km)": round(current_pos, 2),
                "类型": event_type,
                "特征": f"反射峰约 {local_peak:.1f} dB",
                "建议": "检查连接端面、法兰盘或断点"
            })
            i += dead_zone_pts
            continue
        if local_drop < drop_thresh:
            events_found.append({
                "位置 (km)": round(current_pos, 2),
                "类型": "非反射损耗事件（熔接/弯曲）",
                "特征": f"损耗台阶约 {abs(local_drop):.2f} dB",
                "建议": "复测熔接点或检查弯曲半径"
            })
            i += dead_zone_pts
            continue
        i += 1

    return pd.DataFrame(events_found)

class PSO_Optimizer:
    def __init__(self, target_ber, fixed_params, population_size=20, max_iter=30):
        self.target_ber = target_ber
        self.params = fixed_params
        self.pop_size = population_size
        self.max_iter = max_iter
        
        self.bounds_min = np.array([-10.0, 10.0])
        self.bounds_max = np.array([10.0, 30.0])
        
        self.positions = np.random.uniform(self.bounds_min, self.bounds_max, (self.pop_size, 2))
        self.velocities = np.zeros((self.pop_size, 2))
        
        self.pbest_pos = self.positions.copy()
        self.pbest_scores = np.full(self.pop_size, float('inf'))
        self.gbest_pos = None
        self.gbest_score = float('inf')
        self.gbest_result = None
        
        self.history = []

    def fitness_function(self, p_power, p_gain):
        temp_params = self.params.copy()
        temp_params['tx_power'] = p_power
        temp_params['edfa_gain'] = p_gain
        
        res = calculate_simulation(temp_params)
        current_ber = res['ber']
        
        if current_ber > self.target_ber:
            penalty = 1000 + (np.log10(current_ber) - np.log10(self.target_ber)) * 1000
        else:
            penalty = 0
            
        norm_p = (p_power - (-10)) / 20.0
        norm_g = (p_gain - 10) / 20.0
        
        cost = penalty + (0.5 * norm_p + 0.5 * norm_g)
        
        return cost, res

    def run(self):
        w = 0.5
        c1 = 1.5
        c2 = 1.5

        for i in range(self.max_iter):
            for j in range(self.pop_size):
                self.positions[j] = np.clip(self.positions[j], self.bounds_min, self.bounds_max)
                
                p_power, p_gain = self.positions[j]
                cost, res = self.fitness_function(p_power, p_gain)
                
                if cost < self.pbest_scores[j]:
                    self.pbest_scores[j] = cost
                    self.pbest_pos[j] = self.positions[j].copy()
                    
                if cost < self.gbest_score:
                    self.gbest_score = cost
                    self.gbest_pos = self.positions[j].copy()
                    self.gbest_result = res
            
            r1 = np.random.rand(self.pop_size, 2)
            r2 = np.random.rand(self.pop_size, 2)
            
            self.velocities = (w * self.velocities + 
                               c1 * r1 * (self.pbest_pos - self.positions) + 
                               c2 * r2 * (self.gbest_pos - self.positions))
            
            self.positions += self.velocities
            
            self.history.append(self.gbest_score)
            
        return self.gbest_pos, self.gbest_result, self.history

def optimize_system_parameters_pso(target_ber, fixed_params):
    optimizer = PSO_Optimizer(target_ber, fixed_params)
    best_pos, best_res, history = optimizer.run()
    success = (best_res['ber'] <= target_ber)
    return success, best_pos, best_res, history

# ==========================================
# 1. 基础配置与UI专业化设置
# ==========================================
st.set_page_config(
    page_title="智光实验室 | Intelligent Optical Lab",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 10

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e5e7eb; font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif; }
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #334155; }
    h1, h2, h3 { font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif; color: #f8fafc; font-weight: 650; }
    h1 { letter-spacing: 0.3px; }
    p, li, label, div { font-size: 0.98rem; }
    div[data-testid="stMetric"] { background-color: #1e293b; border: 1px solid #334155; padding: 14px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.22); text-align: center; transition: all 0.2s ease; }
    div[data-testid="stMetric"]:hover { border-color: #38bdf8; box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.25); }
    div[data-testid="stMetricLabel"] { color: #cbd5e1 !important; font-size: 0.92rem; }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-family: "Roboto Mono", "Consolas", monospace; font-weight: 700; }
    .device-box { border: 1px solid #334155; padding: 20px; border-radius: 12px; background: linear-gradient(145deg, #1e293b, #111827); box-shadow: inset 1px 1px 3px #020617; margin-bottom: 15px; text-align: center; }
    .lcd-display { background-color: #020617; border: 1px solid #475569; border-radius: 6px; padding: 10px; margin-bottom: 5px; font-family: 'Consolas', 'Courier New', monospace; letter-spacing: 1px; }
    div[data-testid="stExpander"] { background-color: #111827; border-radius: 10px; border: 1px solid #334155; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.02rem; font-weight: 600; color: #93c5fd; }
    .stButton > button { width: 100%; border-radius: 7px; font-weight: 600; }
    /* 顶部实验模块导航修复：模块较多时自动换行，避免右侧“智能导师”等标签被挤出屏幕且无法点击 */
    .stTabs [data-baseweb="tab-list"], div[role="tablist"] {
        gap: 8px;
        flex-wrap: wrap !important;
        overflow-x: visible !important;
        overflow-y: visible !important;
        height: auto !important;
        row-gap: 8px;
    }
    .stTabs [data-baseweb="tab"], div[role="tab"] {
        background-color: #111827;
        border-radius: 6px 6px 0 0;
        padding: 10px 18px;
        color: #cbd5e1;
        flex: 0 0 auto !important;
        white-space: nowrap !important;
    }
    .stTabs [aria-selected="true"] { background-color: #1e293b !important; color: #38bdf8 !important; border-bottom: 2px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)

if 'ber_history' not in st.session_state:
    st.session_state.ber_history = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": """
        你是由湖南科技大学光电实验室开发的“智光AI导师”。
        你的核心任务是执行“四元赋能，双链协同”的教学改革理念：
        1. **理论辅导**：解释光通信原理（如色散、非线性效应）。
        2. **实验指导**：当学生调整参数不合理时（如入纤功率>10dBm），给出预警。
        3. **课程思政**：在回答中适当融入科学精神、大国工匠精神（如赵梓森院士的事迹）。
        4. **结果评价**：根据学生的误码率结果，给出“优/良/中/差”的评级。
        请保持学术严谨，同时语言通俗易懂。
        """},
        {"role": "assistant", "content": "同学你好！我是你的AI实验导师。请在左侧设置实验参数，我可以帮你分析链路性能或生成实验报告。"}
    ]

if 'random_bits' not in st.session_state:
    st.session_state.random_bits = [random.randint(0, 1) for _ in range(20)]

# ==========================================
# 3. 侧边栏控制 (全局状态)
# ==========================================
with st.sidebar:
    st.markdown("## 📡 智光实验室")
    st.title("⚙️ 实验参数设置")

    st.markdown("### 🤖 AI 模式设置")
    ai_mode = st.selectbox(
        "AI 运行模式",
        ["规则库模式（云端推荐）", "在线 API 模式", "本地 LM Studio 模式"],
        index=0,
        help="云端部署建议使用规则库模式；需要大模型时可配置在线 API；本地运行时可连接 LM Studio。"
    )

    client = None
    active_model_id = DEFAULT_MODEL_FALLBACK
    ai_engine_status = "规则库模式（云端可用）"

    if ai_mode == "在线 API 模式":
        api_key = get_secret_or_env("OPENAI_API_KEY", "")
        base_url = get_secret_or_env("OPENAI_BASE_URL", "https://api.openai.com/v1")
        online_model = get_secret_or_env("OPENAI_MODEL", "gpt-4o-mini")
        active_model_id = online_model
        if OPENAI_AVAILABLE and api_key:
            try:
                client = OpenAI(base_url=base_url, api_key=api_key)
                ai_engine_status = f"在线 API ({active_model_id})"
                st.success("✅ 在线 API 已配置")
            except Exception as exc:
                AI_CLIENT_ERROR = f"在线 API 初始化失败：{exc}"
                st.warning("⚠️ 在线 API 初始化失败，已降级为规则库模式")
        else:
            AI_CLIENT_ERROR = "未配置 OPENAI_API_KEY 或缺少 openai 依赖"
            st.warning("⚠️ 未配置在线 API，已降级为规则库模式")
            st.caption("在 Streamlit Secrets 中配置 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL 后可启用。")

    elif ai_mode == "本地 LM Studio 模式":
        st.caption("本模式仅适合本地运行；云端部署无法访问你电脑上的 localhost。")
        lm_base_url = st.text_input("AI 服务地址", value=DEFAULT_LM_STUDIO_BASE_URL, key="lm_base_url")
        manual_model_id = st.text_input("模型 ID（可选，留空自动探测）", value="", key="lm_model_override", placeholder="例如：qwen3-8b")
        active_model_id = manual_model_id.strip() or DEFAULT_MODEL_FALLBACK
        if OPENAI_AVAILABLE:
            try:
                detected_model_ids, detect_error = fetch_local_model_ids(lm_base_url, DEFAULT_LM_STUDIO_API_KEY)
                if detected_model_ids:
                    active_model_id = manual_model_id.strip() or detected_model_ids[0]
                    client = OpenAI(base_url=lm_base_url, api_key=DEFAULT_LM_STUDIO_API_KEY)
                    ai_engine_status = f"Local LM Studio ({active_model_id})"
                    st.success(f"✅ 已连接 LM Studio，检测到 {len(detected_model_ids)} 个模型")
                else:
                    AI_CLIENT_ERROR = f"LM Studio 未连接或未加载模型：{detect_error or '未检测到模型'}"
                    st.warning("⚠️ LM Studio 未连接，已降级为规则库模式")
            except Exception as exc:
                AI_CLIENT_ERROR = f"本地 AI 客户端初始化失败：{exc}"
                st.warning("⚠️ 本地 AI 初始化失败，已降级为规则库模式")
        else:
            AI_CLIENT_ERROR = f"未安装 openai 依赖：{OPENAI_IMPORT_ERROR}"
            st.warning("⚠️ 缺少 openai 依赖，已降级为规则库模式")
    else:
        st.success("✅ 当前使用规则库模式：无需 API Key，适合云端公开部署")

    if 'opt_power' not in st.session_state:
        st.session_state.opt_power = 0.0
    if 'opt_gain' not in st.session_state:
        st.session_state.opt_gain = 15.0 

    st.markdown("### 🌐 光源设置")
    wavelength = st.selectbox("工作波长", ["1550 nm (C-Band)", "1310 nm (O-Band)"])
    
    default_att = 0.20 if "1550" in wavelength else 0.35
    default_disp = 17 if "1550" in wavelength else 0
    
    st.markdown("### 1. 发射机 (Transmitter)")
    mod_options = ["OOK", "FSK", "DPSK", "PAM4", "QPSK", "8-PSK", "16-QAM"]
    mod_format = st.selectbox("调制格式 (Tab 1)", mod_options, index=3)
    
    def update_power():
        st.session_state.opt_power = st.session_state.tx_power_slider

    tx_power = st.slider("发射功率 (dBm)", -10.0, 10.0, value=float(st.session_state.opt_power), step=0.5, key="tx_power_slider", on_change=update_power)

    # 【新增】发射机高级参数：类似 OptiSystem 的 CW Laser / MZM / Directly Modulated Laser 非理想项。
    with st.expander("🚀 发射机非理想参数（OptiSystem对齐）", expanded=False):
        tx_impair_enabled = st.checkbox("启用发射机非理想影响", value=False, key="tx_impair_enabled")
        tx_ext_ratio_db = st.slider("消光比 ER (dB)", 3.0, 20.0, 12.0, 0.5, key="tx_ext_ratio_db",
                                    help="ER 越低，0 码泄漏越明显，眼图垂直开口变小。")
        mzm_vpi = st.slider("MZM 半波电压 Vπ (V)", 1.0, 8.0, 4.0, 0.1, key="mzm_vpi")
        mzm_bias_v = st.slider("MZM 偏置电压 Vbias (V)", 0.0, 8.0, 2.0, 0.1, key="mzm_bias_v",
                               help="最佳教学参考点约为 Vπ/2，偏离后调制线性变差。")
        mzm_vpp = st.slider("MZM 调制电压 Vpp (V)", 0.1, 10.0, 4.0, 0.1, key="mzm_vpp",
                            help="Vpp 太小会调制不足，过大则可能产生非线性过驱动。")
        laser_linewidth_mhz = st.slider("激光器线宽 (MHz)", 0.01, 20.0, 1.0, 0.01, key="laser_linewidth_mhz",
                                        help="线宽主要影响 QPSK/16-QAM 等相干调制星座图相位噪声。")
        tx_rin_db_hz = st.slider("RIN 相对强度噪声 (dB/Hz)", -165.0, -125.0, -150.0, 1.0, key="tx_rin_db_hz",
                                 help="数值越接近 -125 dB/Hz，强度噪声越严重。")
        tx_chirp_alpha = st.slider("发射端啁啾系数 α", -5.0, 5.0, 0.0, 0.1, key="tx_chirp_alpha",
                                   help="直接调制 LD 的啁啾会和光纤色散共同造成脉冲展宽。")
        st.caption("开启后：消光比、MZM 偏置/驱动、RIN、线宽和啁啾会转化为发射端等效代价，直接影响 Q 因子、BER、星座图/眼图开口。")
    
    st.markdown("### 2. 光纤链路 (Fiber Link)")
    distance = st.slider("传输距离 (km)", 10, 200, 80, 5)
    attenuation = st.slider(f"衰减系数 (dB/km) @{wavelength[:4]}", 0.15, 0.50, default_att, 0.01)
    dispersion = st.slider("色散系数 (ps/nm/km)", 0, 20, default_disp, 1)

    # 【新增】光纤高级参数：对应 OptiSystem Optical Fiber 中的 PMD、非线性、熔接/连接器等非理想项。
    with st.expander("🧵 光纤链路高级参数（PMD/非线性/接头熔接）", expanded=False):
        fiber_advanced_enabled = st.checkbox("启用光纤高级非理想影响", value=False, key="fiber_advanced_enabled")
        fiber_pmd_ps_sqrtkm = st.slider("PMD 系数 (ps/√km)", 0.00, 1.00, 0.05, 0.01, key="fiber_pmd_ps_sqrtkm",
                                        help="PMD 越大，不同偏振分量到达时间差越明显，眼宽和 Q 因子下降。")
        fiber_gamma_wkm = st.slider("非线性系数 γ (1/W/km)", 0.1, 3.0, 1.3, 0.1, key="fiber_gamma_wkm",
                                    help="γ 越大，高入纤功率下 SPM/XPM 等非线性代价越明显。")
        fiber_aeff_um2 = st.slider("有效面积 Aeff (μm²)", 20.0, 120.0, 80.0, 1.0, key="fiber_aeff_um2",
                                   help="有效面积越小，光强越高，非线性噪声更明显。")
        fiber_disp_slope = st.slider("色散斜率 S (ps/nm²/km)", 0.00, 0.20, 0.08, 0.01, key="fiber_disp_slope")
        link_connector_count = st.number_input("链路连接器数量", min_value=0, max_value=40, value=0, step=1, key="link_connector_count")
        link_connector_loss_db = st.slider("单个连接器损耗 (dB)", 0.00, 1.00, 0.25, 0.01, key="link_connector_loss_db")
        splice_count = st.number_input("熔接点数量", min_value=0, max_value=80, value=0, step=1, key="splice_count")
        splice_loss_db = st.slider("单个熔接损耗 (dB)", 0.00, 0.30, 0.05, 0.01, key="splice_loss_db")
        span_length_setting_km = st.slider("每跨段长度 (km)", 20.0, 120.0, 80.0, 5.0, key="span_length_setting_km")
        link_margin_db = st.slider("工程链路余量 Margin (dB)", 0.0, 10.0, 3.0, 0.5, key="link_margin_db")
        st.caption("开启后：PMD、非线性、接头/熔接和链路余量会改变接收功率、等效 SNR、眼图开口、Q 因子和 BER。")

    # 【新增】WDM 参数：默认关闭，避免影响单通道教学实验；开启后模拟 MUX/DEMUX 插损、串扰和通道间非线性。
    with st.expander("🌈 WDM 波分复用参数（可选）", expanded=False):
        wdm_enabled = st.checkbox("启用 WDM 多信道影响", value=False, key="wdm_enabled")
        wdm_channels = st.select_slider("信道数", options=[1, 2, 4, 8, 16, 32, 40], value=1, key="wdm_channels")
        wdm_spacing_ghz = st.select_slider("信道间隔 (GHz)", options=[25.0, 50.0, 75.0, 100.0, 200.0], value=100.0, key="wdm_spacing_ghz")
        mux_loss_db = st.slider("MUX 插入损耗 (dB)", 0.0, 6.0, 1.5, 0.1, key="mux_loss_db")
        demux_loss_db = st.slider("DEMUX 插入损耗 (dB)", 0.0, 6.0, 1.5, 0.1, key="demux_loss_db")
        wdm_xtalk_db = st.slider("相邻信道串扰 (dB)", -45.0, -15.0, -30.0, 1.0, key="wdm_xtalk_db",
                                 help="串扰越接近 -15 dB，信道间干扰越严重。")
        st.caption("开启后：信道数增加、间隔变小、串扰变差，会提高非线性和串扰代价，接近 OptiSystem WDM Analyzer 的趋势。")
    
    st.markdown("### 3. 放大器 (EDFA)")
    def update_gain():
        st.session_state.opt_gain = st.session_state.edfa_gain_slider
        
    edfa_gain = st.slider("放大增益 (dB)", 0.0, 40.0, value=float(st.session_state.opt_gain), step=0.5, key="edfa_gain_slider", on_change=update_gain)

    # 【新增】EDFA 高级参数：对应 OptiSystem EDFA 中的 NF、Psat、ASE 带宽和增益平坦度。
    with st.expander("🔋 EDFA 高级参数（NF/Psat/ASE）", expanded=False):
        edfa_advanced_enabled = st.checkbox("启用 EDFA 高级非理想影响", value=False, key="edfa_advanced_enabled")
        edfa_nf_db = st.slider("噪声系数 NF (dB)", 3.0, 9.0, 5.5, 0.1, key="edfa_nf_db",
                               help="NF 越高，ASE 噪声越大，OSNR 和 BER 越差。")
        edfa_psat_dbm = st.slider("饱和输出功率 Psat (dBm)", 5.0, 25.0, 20.0, 0.5, key="edfa_psat_dbm",
                                  help="Psat 越低，高增益/高输入功率时越容易增益压缩。")
        edfa_ase_bw_ghz = st.slider("ASE 等效噪声带宽 (GHz)", 1.0, 100.0, 12.5, 0.5, key="edfa_ase_bw_ghz")
        edfa_gain_flatness_db = st.slider("增益平坦度误差 (dB)", 0.0, 5.0, 0.5, 0.1, key="edfa_gain_flatness_db")
        st.caption("开启后：NF、Psat、ASE 带宽和增益平坦度会改变 OSNR、饱和代价、Q 因子和 BER。")

    # 【新增】接收机滤波/判决/眼图分析参数：对应 OptiSystem Low Pass Filter + BER Analyzer + Eye Diagram Analyzer。
    with st.expander("👁️ 接收端滤波、判决与眼图分析", expanded=False):
        decision_eye_enabled = st.checkbox("启用滤波/判决/眼图影响", value=False, key="decision_eye_enabled")
        rx_filter_bw_ghz = st.slider("低通滤波器带宽 (GHz)", 2.0, 60.0, 18.0, 0.5, key="rx_filter_bw_ghz")
        rx_filter_type = st.selectbox("滤波器类型", ["Bessel", "Gaussian", "Butterworth"], index=0, key="rx_filter_type")
        decision_threshold = st.slider("判决门限（归一化）", 0.10, 0.90, 0.50, 0.01, key="decision_threshold",
                                       help="0.5 附近为二电平 OOK/PAM4 教学默认门限；偏离后误判增加。")
        sampling_offset_ui = st.slider("采样时刻偏移 (UI)", -0.50, 0.50, 0.00, 0.01, key="sampling_offset_ui")
        clock_jitter_ps = st.slider("时钟抖动 Jitter (ps)", 0.0, 30.0, 1.0, 0.5, key="clock_jitter_ps")
        st.caption("开启后：带宽不匹配、门限偏移、采样偏移和抖动会直接压缩眼高/眼宽，并影响 Q 因子和 BER。")

    # 【新增】接收机偏置电压与噪声模块：默认关闭，避免影响原有实验结果；开启后影响链路 BER/Q/OSNR。
    with st.expander("🔬 接收机偏置与噪声（高级）", expanded=False):
        receiver_noise_enabled = st.checkbox("启用接收机偏置/噪声影响", value=False, key="receiver_noise_enabled")
        rx_bias_voltage = st.slider("PIN/APD 等效偏置电压 (V)", 0.0, 15.0, 5.0, 0.1, key="rx_bias_voltage")
        rx_responsivity = st.slider("探测器响应度 R (A/W)", 0.2, 1.2, 0.85, 0.05, key="rx_responsivity")
        rx_bandwidth_ghz = st.slider("接收机等效带宽 (GHz)", 1.0, 50.0, 12.5, 0.5, key="rx_bandwidth_ghz")
        rx_dark_current_na = st.number_input("暗电流 Id (nA)", min_value=0.0, max_value=1000.0, value=2.0, step=0.5, key="rx_dark_current_na")
        rx_load_ohm = st.select_slider("负载电阻 RL (Ω)", options=[25.0, 50.0, 75.0, 100.0, 200.0, 500.0], value=50.0, key="rx_load_ohm")
        rx_noise_scale = st.slider("噪声强度系数", 0.0, 5.0, 1.0, 0.1, key="rx_noise_scale")
        st.caption("开启后：低偏置会降低响应度，高偏置会增加暗电流；带宽越大，热噪声和散粒噪声越明显。")
    
    # 编码设置已集成到“🔣 线路编码”实验页，侧边栏仅保留全局光链路参数。

params = {
    "modulation": mod_format,
    "tx_power": tx_power,
    "distance": distance,
    "attenuation": attenuation,
    "dispersion": dispersion,
    "edfa_gain": edfa_gain,
    "wavelength": wavelength,
    "fiber_advanced_enabled": fiber_advanced_enabled,
    "fiber_pmd_ps_sqrtkm": fiber_pmd_ps_sqrtkm,
    "fiber_gamma_wkm": fiber_gamma_wkm,
    "fiber_aeff_um2": fiber_aeff_um2,
    "fiber_disp_slope": fiber_disp_slope,
    "link_connector_count": link_connector_count,
    "link_connector_loss_db": link_connector_loss_db,
    "splice_count": splice_count,
    "splice_loss_db": splice_loss_db,
    "span_length_setting_km": span_length_setting_km,
    "link_margin_db": link_margin_db,
    "edfa_advanced_enabled": edfa_advanced_enabled,
    "edfa_nf_db": edfa_nf_db,
    "edfa_psat_dbm": edfa_psat_dbm,
    "edfa_ase_bw_ghz": edfa_ase_bw_ghz,
    "edfa_gain_flatness_db": edfa_gain_flatness_db,
    "decision_eye_enabled": decision_eye_enabled,
    "rx_filter_bw_ghz": rx_filter_bw_ghz,
    "rx_filter_type": rx_filter_type,
    "decision_threshold": decision_threshold,
    "sampling_offset_ui": sampling_offset_ui,
    "clock_jitter_ps": clock_jitter_ps,
    "wdm_enabled": wdm_enabled,
    "wdm_channels": wdm_channels,
    "wdm_spacing_ghz": wdm_spacing_ghz,
    "mux_loss_db": mux_loss_db,
    "demux_loss_db": demux_loss_db,
    "wdm_xtalk_db": wdm_xtalk_db,
    "tx_impair_enabled": tx_impair_enabled,
    "tx_ext_ratio_db": tx_ext_ratio_db,
    "mzm_bias_v": mzm_bias_v,
    "mzm_vpi": mzm_vpi,
    "mzm_vpp": mzm_vpp,
    "laser_linewidth_mhz": laser_linewidth_mhz,
    "tx_rin_db_hz": tx_rin_db_hz,
    "tx_chirp_alpha": tx_chirp_alpha,
    "receiver_noise_enabled": receiver_noise_enabled,
    "rx_bias_voltage": rx_bias_voltage,
    "rx_responsivity": rx_responsivity,
    "rx_bandwidth_ghz": rx_bandwidth_ghz,
    "rx_dark_current_na": rx_dark_current_na,
    "rx_load_ohm": rx_load_ohm,
    "rx_noise_scale": rx_noise_scale
}

# 全局计算，由于加入了 cache_data，只有侧边栏参数变化才会重新执行
metrics = calculate_simulation(params)
current_ber_log = math.log10(metrics['ber'])
if not st.session_state.ber_history or st.session_state.ber_history[-1] != current_ber_log:
    st.session_state.ber_history.append(current_ber_log)
    if len(st.session_state.ber_history) > 50: st.session_state.ber_history.pop(0)

st.sidebar.markdown("---")
if st.sidebar.button("📝 生成实验报告"):
    report_prompt = f"""
    请为学生生成一份《光纤通信系统仿真实验报告》。
    **实验参数**: 波长: {params['wavelength']}, 距离: {params['distance']} km, 发射功率: {params['tx_power']} dBm, 调制格式: {params['modulation']}
    **实验结果**: BER: {metrics['ber']:.2e}, Q因子: {metrics['q_factor']:.2f}, OSNR: {metrics['osnr']:.2f} dB, 接收光功率: {metrics['rx_power']:.2f} dBm
    **要求**:
    1. 按照“实验目的”、“实验数据分析”、“存在问题与改进建议”、“最终评分”四个部分撰写。
    2. 如果 BER > 1e-3，指出参数设置不合理之处。
    3. 结合“四元赋能”理念，给出一个简短的思政评语。
    """
    if client is None:
        report_content = generate_rule_based_report(params, metrics)
        show_report_dialog(report_content, "🎓 云端规则库实验报告")
    else:
        with st.spinner("AI 导师正在综合您的实验数据，生成评估报告..."):
            try:
                completion = client.chat.completions.create(
                    model=active_model_id,
                    messages=[{"role": "user", "content": report_prompt}],
                    temperature=0.7,
                )
                report_content = completion.choices[0].message.content
            except Exception as e:
                st.warning(f"大模型生成失败，已自动切换到规则库报告：{e}")
                report_content = generate_rule_based_report(params, metrics)
            show_report_dialog(report_content)

# ==========================================
# 4. 主界面内容框架
# ==========================================
st.markdown("# 🔬 智光实验室：光通信AI仿真平台")
st.caption(f"当前工作波长: **{wavelength}** | 模式: 综合实验 | AI 引擎: **{ai_engine_status}**")

tab1, tab5, tab_ld, tab_fiber_param, tab_fiber_principle, tab_edfa, tab_otdr, tab2, tab4, tab_phone, tab_design, tab3 = st.tabs([
    "📈 链路仿真", "🛠️ 器件测试", "💡 LD光源", "🔀 光纤参数", "🧭 传输原理", "🔋 EDFA仿真", 
    "📉 OTDR实验", "🤖 AI 诊断", "🔣 线路编码", "📞 电话系统", "🏆 综合设计", "🎓 智能导师"
])

# ==========================================
# 5. 独立模块封装区域 (@st.fragment)
# ==========================================

@st.fragment
def render_tab1_link_sim(params, metrics, client, active_model_id):
    st.markdown("### 📈 仿真模式选择")
    sim_mode = st.radio("选择仿真内核", ["⚡ 快速教学模式 (Analytical Model)", "🐢 高保真科研模式 (IFTS-PyTorch)"], horizontal=True)

    if sim_mode == "⚡ 快速教学模式 (Analytical Model)":
        st.info("当前快速模式采用工程近似模型：链路预算 + EDFA ASE + 简化 GN 非线性噪声 + 色散功率代价 + 调制格式 BER 近似。结果用于教学与工程趋势分析，不等同于商用软件或完整 SSFM 高保真仿真。")
        st.markdown("### 📊 核心传输指标 (Key Metrics)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📡 接收光功率", f"{metrics['rx_power']:.2f} dBm")
        col2.metric("📶 OSNR 估计", f"{metrics['osnr']:.2f} dB")
        col3.metric("✨ Q因子/判决裕量", f"{metrics['q_factor']:.2f}")
        ber_val = metrics['ber']
        col4.metric("📉 误码率 BER", f"{ber_val:.2e}", delta_color="inverse" if ber_val > 1e-3 else "normal")
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("接收灵敏度", f"{metrics.get('rx_sensitivity_dbm', -99):.1f} dBm")
        col6.metric("灵敏度余量", f"{metrics.get('sensitivity_margin_db', 0):.2f} dB")
        col7.metric("色散代价", f"{metrics.get('dispersion_penalty_db', 0):.2f} dB")
        col8.metric("EDFA 跨段数", f"{metrics.get('num_spans', 1)}")

        # 【新增】OptiSystem 对齐高级模块的直观结果展示。
        advanced_enabled_any = any([
            metrics.get("fiber_advanced_enabled", False), metrics.get("edfa_advanced_enabled", False),
            metrics.get("decision_eye_enabled", False), metrics.get("wdm_enabled", False)
        ])
        if advanced_enabled_any:
            st.markdown("#### 🧩 OptiSystem 对齐高级模块影响总览")
            adv_cols = st.columns(4)
            adv_cols[0].metric("链路高级代价", f"{metrics.get('link_advanced_penalty_db', 0):.2f} dB")
            adv_cols[1].metric("判决/眼图代价", f"{metrics.get('analysis_penalty_db', 0):.2f} dB")
            adv_cols[2].metric("MUX/DEMUX 损耗", f"{metrics.get('mux_demux_loss_db', 0):.2f} dB")
            adv_cols[3].metric("ASE 功率", f"{metrics.get('ase_power_dbm', -100):.1f} dBm")

            adv_rows = []
            if metrics.get("fiber_advanced_enabled", False):
                adv_rows.extend([
                    ["PMD 偏振模色散", f"DGD≈{metrics.get('pmd_dgd_ps', 0):.2f} ps", f"{metrics.get('fiber_pmd_penalty_db', 0):.2f} dB", "眼宽减小，采样容限下降"],
                    ["非线性 SPM/XPM", f"γ={metrics.get('fiber_gamma_wkm', 0):.2f}, Aeff={metrics.get('fiber_aeff_um2', 0):.0f} μm²", f"{metrics.get('fiber_nli_penalty_db', 0):.2f} dB", "高发射功率时 BER 更容易变差"],
                    ["接头/熔接/余量", f"连接器={metrics.get('link_connector_total_loss_db', 0):.2f} dB，熔接={metrics.get('splice_total_loss_db', 0):.2f} dB，余量={metrics.get('link_margin_penalty_db', 0):.2f} dB", "影响接收功率", "接收功率下降，灵敏度余量变小"],
                ])
            if metrics.get("edfa_advanced_enabled", False):
                adv_rows.extend([
                    ["EDFA 噪声系数 NF", f"NF={metrics.get('edfa_nf_db', 0):.1f} dB", "影响 OSNR", "NF 增大 → ASE 增大 → OSNR/Q 下降"],
                    ["EDFA 饱和/平坦度", f"Psat={metrics.get('edfa_psat_dbm', 0):.1f} dBm，平坦度代价={metrics.get('edfa_flatness_penalty_db', 0):.2f} dB", f"饱和代价={metrics.get('gain_saturation_penalty_db', 0):.2f} dB", "增益过高不一定更好"],
                ])
            if metrics.get("wdm_enabled", False):
                adv_rows.extend([
                    ["WDM 插损与串扰", f"{metrics.get('wdm_channels', 1)} 通道，间隔={metrics.get('wdm_spacing_ghz', 0):.0f} GHz", f"串扰={metrics.get('wdm_xtalk_penalty_db', 0):.2f} dB，间隔={metrics.get('wdm_spacing_penalty_db', 0):.2f} dB", "信道越多、间隔越小，串扰和非线性越明显"],
                ])
            if metrics.get("decision_eye_enabled", False):
                adv_rows.extend([
                    ["接收滤波器", f"{metrics.get('rx_filter_type', '')}, BW={metrics.get('rx_filter_bw_ghz', 0):.1f} GHz", f"{metrics.get('decision_filter_penalty_db', 0):.2f} dB", "带宽太窄产生 ISI，太宽引入更多噪声"],
                    ["判决与采样", f"门限={metrics.get('decision_threshold', 0):.2f}, 偏移={metrics.get('sampling_offset_ui', 0):+.2f} UI, 抖动={metrics.get('clock_jitter_ps', 0):.1f} ps", f"{metrics.get('decision_threshold_penalty_db', 0)+metrics.get('sampling_jitter_penalty_db', 0):.2f} dB", "最佳采样点偏移会导致眼图闭合"],
                ])
            if adv_rows:
                st.dataframe(pd.DataFrame(adv_rows, columns=["模块", "当前参数", "等效影响", "学生观察重点"]), use_container_width=True, hide_index=True)

                labels = []
                values = []
                for label, key in [("PMD", "fiber_pmd_penalty_db"), ("非线性", "fiber_nli_penalty_db"), ("色散斜率", "fiber_slope_penalty_db"), ("EDFA平坦度", "edfa_flatness_penalty_db"), ("WDM串扰", "wdm_xtalk_penalty_db"), ("WDM间隔", "wdm_spacing_penalty_db"), ("滤波", "decision_filter_penalty_db"), ("门限", "decision_threshold_penalty_db"), ("采样/抖动", "sampling_jitter_penalty_db")]:
                    val = metrics.get(key, 0)
                    if val > 0.01:
                        labels.append(label); values.append(val)
                if values:
                    fig_adv, ax_adv = plt.subplots(figsize=(8, 2.8))
                    ax_adv.bar(labels, values)
                    ax_adv.set_ylabel("等效代价 (dB)")
                    ax_adv.set_title("高级模块对 Q/BER 的贡献分解")
                    ax_adv.grid(True, alpha=0.25)
                    st.pyplot(fig_adv)

        if metrics.get("decision_eye_enabled", False):
            st.markdown("#### 👁️ Eye Diagram / 眼图分析（BER Analyzer 对齐显示）")
            eye_t, eye_segments, eye_info = generate_eye_diagram_data(metrics, params)
            ecols = st.columns(4)
            ecols[0].metric("眼高 Eye Height", f"{eye_info['eye_height']:.2f}")
            ecols[1].metric("眼宽 Eye Width", f"{eye_info['eye_width']:.2f} UI")
            ecols[2].metric("噪声裕量", f"{eye_info['noise_margin']:.2f}")
            ecols[3].metric("等效抖动", f"{eye_info['jitter_ui']:.2f} UI")
            fig_eye, ax_eye = plt.subplots(figsize=(8, 3.2))
            for seg in eye_segments[:70]:
                ax_eye.plot(eye_t, seg, alpha=0.18, linewidth=0.8)
            ax_eye.axvline(0, linestyle="--", linewidth=1.0, alpha=0.7, label="采样中心")
            ax_eye.axhline(2*metrics.get('decision_threshold', 0.5)-1, linestyle=":", linewidth=1.2, alpha=0.8, label="判决门限")
            ax_eye.set_xlabel("Time (UI)")
            ax_eye.set_ylabel("Normalized Amplitude")
            ax_eye.set_title("教学眼图：参数越差，眼高/眼宽越小，BER 越高")
            ax_eye.grid(True, alpha=0.25)
            ax_eye.legend(loc="best")
            st.pyplot(fig_eye)
            st.caption("眼图与 OptiSystem 的 Eye Diagram Analyzer 趋势对齐：噪声主要压缩眼高，色散/PMD/滤波/抖动主要压缩眼宽。")

        if metrics.get("tx_impair_enabled", False):
            tx1, tx2, tx3, tx4 = st.columns(4)
            tx1.metric("发射端总代价", f"{metrics.get('tx_total_penalty_db', 0):.2f} dB")
            tx2.metric("消光比 ER", f"{metrics.get('tx_ext_ratio_db', 0):.1f} dB")
            tx3.metric("MZM 调制深度", f"{metrics.get('tx_modulation_depth', 1)*100:.0f}%")
            tx4.metric("啁啾-色散代价", f"{metrics.get('tx_chirp_penalty_db', 0):.2f} dB")
            st.markdown("#### 🚀 发射机非理想影响对比")
            tx_compare_df = pd.DataFrame({
                "发射端因素": ["消光比不足", "MZM 偏置偏离", "MZM 驱动不匹配", "激光器线宽", "RIN 强度噪声", "啁啾 × 色散"],
                "当前参数": [
                    f"ER={metrics.get('tx_ext_ratio_db', 0):.1f} dB",
                    f"Vbias={metrics.get('mzm_bias_v', 0):.1f} V，Vπ={metrics.get('mzm_vpi', 0):.1f} V",
                    f"Vpp={metrics.get('mzm_vpp', 0):.1f} V，深度={metrics.get('tx_modulation_depth', 1)*100:.0f}%",
                    f"{metrics.get('laser_linewidth_mhz', 0):.2f} MHz",
                    f"{metrics.get('tx_rin_db_hz', 0):.0f} dB/Hz",
                    f"α={metrics.get('tx_chirp_alpha', 0):.1f}，累计色散={metrics.get('dispersion_total', 0):.0f} ps/nm",
                ],
                "等效代价": [
                    f"{metrics.get('tx_er_penalty_db', 0):.2f} dB",
                    f"{metrics.get('tx_mzm_bias_penalty_db', 0):.2f} dB",
                    f"{metrics.get('tx_mzm_drive_penalty_db', 0):.2f} dB",
                    f"{metrics.get('tx_linewidth_penalty_db', 0):.2f} dB",
                    f"{metrics.get('tx_rin_penalty_db', 0):.2f} dB",
                    f"{metrics.get('tx_chirp_penalty_db', 0):.2f} dB",
                ],
                "学生观察重点": [
                    "ER 降低会让 0/1 电平差变小，眼图垂直开口减小",
                    "偏置点偏离 Vπ/2 时，调制器不再处在线性工作区",
                    "Vpp 太小调制不足，过大容易过驱动失真",
                    "线宽会带来相位噪声，高阶相干调制更敏感",
                    "RIN 表示激光强度抖动，OOK/PAM4 更敏感",
                    "啁啾和色散共同作用，距离越长越明显",
                ]
            })
            st.dataframe(tx_compare_df, use_container_width=True, hide_index=True)
            penalty_items = [
                metrics.get('tx_er_penalty_db', 0), metrics.get('tx_mzm_bias_penalty_db', 0),
                metrics.get('tx_mzm_drive_penalty_db', 0), metrics.get('tx_linewidth_penalty_db', 0),
                metrics.get('tx_rin_penalty_db', 0), metrics.get('tx_chirp_penalty_db', 0)
            ]
            fig_tx_pen, ax_tx_pen = plt.subplots(figsize=(8, 2.8))
            ax_tx_pen.bar(["ER", "偏置", "Vpp", "线宽", "RIN", "啁啾"], penalty_items)
            ax_tx_pen.set_ylabel("等效代价 (dB)")
            ax_tx_pen.set_title("发射端非理想因素对链路质量的贡献")
            ax_tx_pen.grid(True, alpha=0.25)
            st.pyplot(fig_tx_pen)
            st.caption("这些发射端参数主要改变调制质量和等效 SNR，因此最直观的变化在 Q 因子、BER、星座图散布和发射端总代价中。")

        if metrics.get("receiver_noise_enabled", False):
            nr1, nr2, nr3, nr4 = st.columns(4)
            nr1.metric("接收机偏置", f"{metrics.get('rx_bias_voltage', 0):.1f} V")
            nr2.metric("偏置代价", f"{metrics.get('rx_bias_penalty_db', 0):.2f} dB")
            nr3.metric("等效暗电流", f"{metrics.get('rx_dark_current_na', 0):.2f} nA")
            nr4.metric("热/散粒噪声", f"{metrics.get('rx_thermal_noise_dbm', -100):.1f}/{metrics.get('rx_shot_noise_dbm', -100):.1f} dBm")
            st.markdown("#### 🔎 接收机非理想影响对比")
            compare_df = pd.DataFrame({
                "指标": ["Q 因子", "BER", "等效 SNR", "接收灵敏度"],
                "原模型": [
                    f"{metrics.get('base_q_factor', metrics['q_factor']):.2f}",
                    f"{metrics.get('base_ber', metrics['ber']):.2e}",
                    f"{metrics.get('base_effective_snr_db', 0):.2f} dB",
                    f"{metrics.get('base_rx_sensitivity_dbm', metrics.get('rx_sensitivity_dbm', 0)):.1f} dBm",
                ],
                "加入接收机偏置/噪声后": [
                    f"{metrics['q_factor']:.2f}",
                    f"{metrics['ber']:.2e}",
                    f"{metrics.get('effective_snr_db', 0):.2f} dB",
                    f"{metrics.get('rx_sensitivity_dbm', 0):.1f} dBm",
                ],
                "变化": [
                    f"{metrics['q_factor'] - metrics.get('base_q_factor', metrics['q_factor']):+.2f}",
                    "变差" if metrics['ber'] > metrics.get('base_ber', metrics['ber']) else "基本不变",
                    f"{metrics.get('effective_snr_db', 0) - metrics.get('base_effective_snr_db', 0):+.2f} dB",
                    f"{metrics.get('rx_sensitivity_dbm', 0) - metrics.get('base_rx_sensitivity_dbm', metrics.get('rx_sensitivity_dbm', 0)):+.2f} dB",
                ]
            })
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
            st.caption(f"接收机等效总代价：{metrics.get('rx_total_penalty_db', 0):.2f} dB。OSNR 主要是光链路指标，所以接收机参数主要改变 Q 因子、BER 和灵敏度，而不是直接改变 OSNR。")

        st.markdown("#### 🧪 实验对比记录（用于论文/课堂分析）")
        with st.expander("保存和比较当前参数组", expanded=False):
            label_default = f"{params.get('modulation','')}-{params.get('distance',0)}km-{params.get('tx_power',0)}dBm"
            exp_label = st.text_input("实验组名称", value=label_default, key="exp_compare_label")
            c_save, c_clear = st.columns(2)
            if c_save.button("💾 保存当前实验结果", key="save_exp_compare"):
                if 'exp_compare_records' not in st.session_state:
                    st.session_state.exp_compare_records = []
                record = {
                    "实验组": exp_label,
                    "调制": params.get('modulation'),
                    "距离(km)": params.get('distance'),
                    "Tx功率(dBm)": params.get('tx_power'),
                    "Rx功率(dBm)": round(metrics.get('rx_power', 0), 2),
                    "OSNR(dB)": round(metrics.get('osnr', 0), 2),
                    "Q因子": round(metrics.get('q_factor', 0), 2),
                    "BER": f"{metrics.get('ber', 1):.2e}",
                    "发射端代价(dB)": round(metrics.get('tx_total_penalty_db', 0), 2),
                    "链路高级代价(dB)": round(metrics.get('link_advanced_penalty_db', 0), 2),
                    "接收/判决代价(dB)": round(metrics.get('rx_total_penalty_db', 0)+metrics.get('analysis_penalty_db', 0), 2),
                    "主要退化原因": max([
                        (metrics.get('tx_total_penalty_db', 0), '发射机非理想'),
                        (metrics.get('link_advanced_penalty_db', 0), '光纤/EDFA/WDM 链路代价'),
                        (metrics.get('rx_total_penalty_db', 0), '接收机噪声'),
                        (metrics.get('analysis_penalty_db', 0), '滤波/判决/眼图采样'),
                        (metrics.get('dispersion_penalty_db', 0), '色散')
                    ], key=lambda x: x[0])[1]
                }
                st.session_state.exp_compare_records.append(record)
                st.success("已保存当前实验组，可继续调整参数后保存下一组。")
            if c_clear.button("🧹 清空对比记录", key="clear_exp_compare"):
                st.session_state.exp_compare_records = []
                st.info("已清空。")
            if st.session_state.get('exp_compare_records'):
                df_cmp = pd.DataFrame(st.session_state.exp_compare_records)
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)
                try:
                    fig_cmp, ax_cmp = plt.subplots(figsize=(8, 2.8))
                    ax_cmp.plot(range(len(df_cmp)), [float(x) for x in df_cmp['Q因子']], marker='o', label='Q因子')
                    ax_cmp.set_xticks(range(len(df_cmp)))
                    ax_cmp.set_xticklabels(df_cmp['实验组'], rotation=20, ha='right')
                    ax_cmp.set_ylabel("Q 因子")
                    ax_cmp.set_title("不同实验组 Q 因子对比")
                    ax_cmp.grid(True, alpha=0.25)
                    ax_cmp.legend(loc='best')
                    st.pyplot(fig_cmp)
                except Exception:
                    pass
                st.download_button("📥 下载实验对比表 CSV", df_cmp.to_csv(index=False).encode('utf-8-sig'), "optisystem_compare.csv", "text/csv")

        st.markdown("### 🖥️ 相干接收机 DSP 概念可视化")
        st.caption("用于展示色散补偿、载波相位恢复等 DSP 环节的趋势；该星座图为概念演示，不替代完整相干接收机 DSP 算法。")
        
        curr_q = metrics['q_factor']
        curr_disp = metrics['dispersion_total']
        pn_std = 0.05 
        s1, s2, s3 = run_coherent_dsp_demo(params['modulation'], curr_q, curr_disp, pn_std)
        
        col_dsp1, col_dsp2, col_dsp3 = st.columns(3)
        def plot_dsp_step(ax, signal, title, color_code):
            sig_show = signal[:500] if len(signal) > 500 else signal
            ax.scatter(sig_show.real, sig_show.imag, s=5, c=color_code, alpha=0.6, edgecolors='none')
            limit = 3.0 if '16-QAM' in params['modulation'] else 2.5
            ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
            ax.set_xticks([]); ax.set_yticks([]) 
            ax.set_title(title, color='#e0e0e0', fontsize=10, pad=10)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
            ax.axvline(0, color='gray', linestyle=':', alpha=0.3)
            ax.set_facecolor('black')

        with col_dsp1:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            plot_dsp_step(ax1, s1, "1. ADC Raw Input\n(Dispersion + Phase Noise)", '#ff4b4b') 
            st.pyplot(fig1)
            st.info("📉 **Rx Raw**: 信号受色散污染，星座图模糊。")

        with col_dsp2:
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            plot_dsp_step(ax2, s2, "2. After CDC\n(Dispersion Removed)", '#ffa500') 
            st.pyplot(fig2)
            st.warning("🔄 **After CDC**: 色散已均衡。因相位噪声旋转。")

        with col_dsp3:
            fig3, ax3 = plt.subplots(figsize=(3, 3))
            plot_dsp_step(ax3, s3, "3. After CPR & MIMO\n(Final Recovered)", '#00ff00') 
            st.pyplot(fig3)
            st.success("✅ **Final Output**: 载波相位锁定，星座图清晰。")

        st.markdown("---")
        st.markdown("### ⏱️ 光时域脉冲展宽与带宽评估 (Optical Time Domain Visualizer)")
        col_pb1, col_pb2 = st.columns([1, 2])
        
        with col_pb1:
            st.markdown("#### 1. 测试信号发生器")
            bit_seq_input = st.text_input("自定义比特序列 (0/1)", "0000000100000000")
            bit_list = [int(b) for b in bit_seq_input if b in ['0', '1']]
            if not bit_list: bit_list = [1] 
                
            t0 = st.number_input("映射的高斯脉冲半宽 T0 (ps)", value=20.0, step=5.0)
            lambda_nm = 1550 if "1550" in params['wavelength'] else 1310
            c_nm_ps = 3e5 
            beta2 = abs((params['dispersion'] * lambda_nm**2) / (2 * np.pi * c_nm_ps))
            
            l_d = (t0 ** 2) / (beta2 + 1e-12)
            t1 = t0 * np.sqrt(1 + (params['distance'] / l_d)**2)
            
            sig_in, sig_out = t0 / np.sqrt(2), t1 / np.sqrt(2)
            delta_sig = np.sqrt(abs(sig_out**2 - sig_in**2))
            f3db = 0.187 / (delta_sig * 1e-12) / 1e9 if delta_sig > 0 else float('inf')
            
            st.markdown("#### 2. 测量结果")
            st.metric("出射脉冲半宽 (T1)", f"{t1:.2f} ps")
            st.metric("3dB 光纤带宽", f"{f3db:.2f} GHz" if f3db != float('inf') else "∞ (无色散)")
            st.caption("该结果为脉冲展宽反推带宽，适合解释色散导致的时域展宽。")
            
        with col_pb2:
            bit_period = 100.0  
            t_ax = np.linspace(-bit_period, bit_period * len(bit_list), max(1000, len(bit_list)*100))
            p_in_total = np.zeros_like(t_ax)
            p_out_total = np.zeros_like(t_ax)
            
            for i, bit in enumerate(bit_list):
                if bit == 1:
                    center_t = i * bit_period
                    p_in_total += np.exp(-((t_ax - center_t)**2) / (t0**2))
                    p_out_total += (t0/t1) * np.exp(-((t_ax - center_t)**2) / (t1**2))
            
            noise_floor = -100.0  
            p_in_dbm = np.maximum(10 * np.log10(p_in_total + 1e-15), noise_floor) 
            p_out_dbm = np.maximum(10 * np.log10(p_out_total + 1e-15), noise_floor)
            
            fig_p, ax_p = plt.subplots(figsize=(8, 3.5))
            ax_p.plot(t_ax, p_in_dbm, color='#00FF00', label='Input (Tx)')
            ax_p.plot(t_ax, p_out_dbm, color='#FF4B4B', label=f'Output (Rx @ {params["distance"]}km)')
            
            for i in range(len(bit_list) + 1):
                ax_p.axvline(i * bit_period - bit_period/2, color='gray', linestyle=':', alpha=0.3)
            
            ax_p.set_xlabel("Time (ps)", color='#e0e0e0')
            ax_p.set_ylabel("Power (dBm)", color='#e0e0e0')
            ax_p.set_title("Optical Time Domain Visualizer (dBm Scale)", color='#e0e0e0')
            ax_p.set_ylim(-105, 5) 
            ax_p.legend(loc="upper right")
            ax_p.grid(True, alpha=0.2)
            st.pyplot(fig_p)

        st.markdown("---")
        st.markdown("### 📡 扫频法 3 dB 带宽测量（OptiSystem 对齐模式）")
        st.caption("模拟 OptiSystem 中 CW Laser + Mach-Zehnder/直接调制 + Optical Fiber + Photodiode/RF Analyzer 的频率响应测试。默认关闭器件带宽限制时，结果主要由色散功率衰落决定。")

        col_bw1, col_bw2 = st.columns([1, 2])
        with col_bw1:
            bw_fmax = st.slider("扫频上限 (GHz)", 5.0, 100.0, 40.0, 5.0, key="rf_bw_fmax")
            bw_points = st.select_slider("扫频点数", options=[101, 201, 401, 801], value=401, key="rf_bw_points")
            include_device_bw = st.checkbox("加入发射机/接收机带宽限制", value=False, key="include_device_bw", help="若要和 OptiSystem 的理想频响实验对齐，建议先关闭；若模拟真实仪器，再开启。")
            if include_device_bw:
                tx_3db_ghz = st.number_input("发射机 3dB 带宽 (GHz)", min_value=1.0, max_value=100.0, value=25.0, step=1.0, key="tx_3db_ghz")
                rx_3db_ghz = st.number_input("接收机 3dB 带宽 (GHz)", min_value=1.0, max_value=100.0, value=25.0, step=1.0, key="rx_3db_ghz")
            else:
                tx_3db_ghz = rx_3db_ghz = 1e9

        with col_bw2:
            f_ghz = np.linspace(0.01, bw_fmax, int(bw_points))
            f_hz = f_ghz * 1e9
            lam_m = 1550e-9 if "1550" in params['wavelength'] else 1310e-9
            c_m_s = 2.998e8
            D_si = float(params['dispersion']) * 1e-6  # ps/nm/km -> s/m^2
            L_m = float(params['distance']) * 1e3

            # IM/DD 双边带调制经过色散后的 RF 功率衰落近似：|cos(pi*D*L*lambda^2*f^2/c)|。
            # 与 OptiSystem 的 Optical Fiber + PIN/RF Analyzer 基础扫频结果趋势一致；若 D=0，则只剩器件带宽限制。
            h_cd = np.abs(np.cos(np.pi * D_si * L_m * (lam_m ** 2) * (f_hz ** 2) / c_m_s))
            if include_device_bw:
                h_tx = 1.0 / np.sqrt(1.0 + (f_ghz / tx_3db_ghz) ** 2)
                h_rx = 1.0 / np.sqrt(1.0 + (f_ghz / rx_3db_ghz) ** 2)
            else:
                h_tx = np.ones_like(f_ghz)
                h_rx = np.ones_like(f_ghz)
            h_total = np.maximum(h_cd * h_tx * h_rx, 1e-12)
            response_db = 20 * np.log10(h_total / h_total[0])

            below = np.where(response_db <= -3.0)[0]
            if len(below) > 0:
                idx = int(below[0])
                if idx > 0:
                    x1, x2 = f_ghz[idx-1], f_ghz[idx]
                    y1, y2 = response_db[idx-1], response_db[idx]
                    f_3db = x1 + (-3.0 - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
                else:
                    f_3db = f_ghz[idx]
                bw_text = f"{f_3db:.2f} GHz"
            else:
                f_3db = None
                bw_text = f"> {bw_fmax:.1f} GHz"

            fig_bw, ax_bw = plt.subplots(figsize=(8, 3.5))
            ax_bw.plot(f_ghz, response_db, linewidth=2.0, label="归一化 RF 响应")
            ax_bw.axhline(-3, linestyle="--", linewidth=1.2, label="-3 dB")
            if f_3db is not None:
                ax_bw.axvline(f_3db, linestyle=":", linewidth=1.2, label=f"3dB 带宽≈{f_3db:.2f} GHz")
            ax_bw.set_xlabel("调制频率 (GHz)")
            ax_bw.set_ylabel("归一化响应 (dB)")
            ax_bw.set_title("扫频法测量光纤/链路 3dB 带宽")
            ax_bw.set_ylim(max(-40, float(np.nanmin(response_db)) - 3), 2)
            ax_bw.grid(True, alpha=0.25)
            ax_bw.legend(loc="best")
            st.pyplot(fig_bw)

            m1, m2, m3 = st.columns(3)
            m1.metric("扫频法 3dB 带宽", bw_text)
            m2.metric("测试距离", f"{params['distance']} km")
            m3.metric("色散系数", f"{params['dispersion']} ps/nm/km")
            st.info("与 OptiSystem 对齐建议：使用 CW Laser → MZM/直接调制 → Optical Fiber → PIN → RF Analyzer，关闭非线性和额外滤波器；本程序关闭器件带宽限制时，主要对齐色散功率衰落趋势。")

        st.markdown("---")
        st.subheader("📉 误码率历史趋势")
        # 即使只有一个 BER 记录点，也绘制图形，避免页面只显示标题没有曲线。
        if len(st.session_state.ber_history) >= 1:
            fig_trend, ax_trend = plt.subplots(figsize=(10, 2.5))
            x_hist = np.arange(len(st.session_state.ber_history))
            y_hist = np.array(st.session_state.ber_history, dtype=float)
            if len(y_hist) == 1:
                ax_trend.scatter(x_hist, y_hist, s=55, label="当前 BER")
                ax_trend.set_xlim(-0.5, 0.5)
                pad = 0.5 if abs(y_hist[0]) < 1 else max(0.5, abs(y_hist[0]) * 0.05)
                ax_trend.set_ylim(y_hist[0] - pad, y_hist[0] + pad)
                st.caption("当前只有一个 BER 记录点；调整侧边栏中的功率、距离、色散、EDFA 增益或调制格式后，会形成历史趋势曲线。")
            else:
                ax_trend.plot(x_hist, y_hist, linewidth=2, marker='o', markersize=3, label="BER 历史")
            ax_trend.set_ylabel("Log10 BER")
            ax_trend.set_xlabel("参数变化次数")
            ax_trend.set_title("误码率历史趋势（BER Evolution Trend）")
            ax_trend.grid(True, alpha=0.3)
            ax_trend.legend(loc="best")
            st.pyplot(fig_trend)
        else:
            st.info("暂无 BER 记录。请先运行一次链路仿真。")

        st.markdown("---")
        st.subheader("🛠️ 逆向设计：光传输系统参数协同优化 (AI-PSO)")
        with st.expander("🎯 启动工程设计模式 (Design Mode)", expanded=False):
            st.markdown(f"**实验任务**：假设你是系统架构师，客户要求在传输 **{params['distance']}km** 时，误码率必须低于 **目标值**。")
            col_opt1, col_opt2 = st.columns([1, 2])
            
            with col_opt1:
                target_ber_exp = st.slider("目标误码率 (1e-X)", 3, 12, 9)
                target_ber = 10 ** (-target_ber_exp)
                st.caption(f"Target BER < {target_ber:.1e}")
                
                if st.button("🚀 启动 PSO 多维协同寻优"):
                    with st.spinner("正在初始化粒子群...进行多参数迭代寻优..."):
                        is_success, best_pos, best_res, hist = optimize_system_parameters_pso(target_ber, params)
                        st.session_state.opt_power = round(best_pos[0], 2)
                        st.session_state.opt_gain = round(best_pos[1], 2)
                        st.session_state.opt_result = best_res
                        st.session_state.opt_history = hist 
                        st.session_state.opt_success = is_success
                        st.rerun() # PSO 修改了全局滑块绑定的状态，需要全页刷新

            with col_opt2:
                if 'opt_history' in st.session_state and 'opt_gain' in st.session_state:
                    hist = st.session_state.opt_history
                    best_p = st.session_state.opt_power
                    best_g = st.session_state.opt_gain
                    
                    fig_opt, ax_opt1 = plt.subplots(figsize=(6, 3))
                    ax_opt1.plot(hist, color='#00FFFF', linewidth=2, marker='o', markersize=3)
                    ax_opt1.set_title("PSO Algorithm Convergence")
                    ax_opt1.set_xlabel("Iterations")
                    ax_opt1.set_ylabel("Cost Function")
                    st.pyplot(fig_opt)

                    if st.session_state.opt_success:
                        st.success("✅ 寻优成功！全局最佳点：")
                        c1, c2 = st.columns(2)
                        c1.metric("最佳发射功率", f"{best_p} dBm")
                        c2.metric("最佳 EDFA 增益", f"{best_g} dB")
                    else:
                        st.error("❌ 寻优失败！无法同时满足 BER 和能耗要求。")
                    
                    if st.button("🎓 解析 PSO 寻优策略"):
                        prompt_design = f"学生使用PSO优化。目标: BER < {target_ber:.1e}。结果: Tx Power = {best_p} dBm, EDFA Gain = {best_g} dB。"
                        if client is None:
                            show_ai_unavailable("PSO 寻优解析")
                        else:
                            try:
                                with st.spinner("AI 导师正在生成算法原理解析..."):
                                    completion = client.chat.completions.create(
                                        model=active_model_id,
                                        messages=[{"role": "user", "content": prompt_design}],
                                    )
                                    st.info(completion.choices[0].message.content)
                            except Exception:
                                st.warning("请连接本地模型以获取详细解释。")

    else:
        st.info("ℹ️ 已切换至基于 PyTorch 的分步傅里叶 (SSFM) + 神经网络 (GAN/BiLSTM) 内核。")
        if not IFTS_AVAILABLE:
            st.warning("🔒 **高级物理内核未解锁 (便携版)**")
            st.markdown("当前运行的是轻量级版本，未包含 `IFTS` 高保真物理仿真依赖包。推荐继续使用 **⚡ 快速教学模式**。")
        else:
            col_ctrl1, col_ctrl2 = st.columns([1, 3])
            with col_ctrl1:
                st.markdown("#### 🚀 仿真控制")
                if st.button("运行深度仿真", type="primary"):
                    st.session_state.run_simulation = True
                
                st.markdown("---")
                st.caption(f"• 发射功率: {params['tx_power']} dBm\n• 传输距离: {params['distance']} km\n• 线宽(模拟): 100 kHz")

            with col_ctrl2:
                if 'run_simulation' in st.session_state and st.session_state.run_simulation:
                    try:
                        with st.spinner("正在初始化 PyTorch 计算图... 执行 SSFM 求解..."):
                            config_path = os.path.join('IFTS', 'simulation_main', 'config', 'paras.yml')
                            if not os.path.exists(config_path):
                                st.error(f"❌ 找不到配置文件: {config_path}")
                                return
                                
                            with open(config_path, 'r', encoding="utf-8") as f:
                                configs = yaml.safe_load(f.read())

                            if 'Sig_Para' not in configs: configs['Sig_Para'] = {}
                            configs['Sig_Para']['nPol'] = 2 
                            configs['Simu_Para']['nPol'] = 2
                            configs['Simu_Para']['channel_num'] = 1 
                            configs['Simu_Para']['sig_power_dbm'] = float(params['tx_power'])
                            configs['Simu_Para']['total_len'] = float(params['distance'])
                            configs['Simu_Para']['span_len'] = float(params['distance'])
                            configs['Simu_Para']['span_num'] = 1
                            if 'Ch_Para' in configs and 'fiber_config' in configs['Ch_Para']:
                                configs['Ch_Para']['fiber_config']['alpha_inndB'] = float(params['attenuation'])
                                configs['Ch_Para']['fiber_config']['D'] = float(params['dispersion'])
                            configs['Simu_Para']['device'] = 'cpu'
                            configs['Simu_Para']['caclu_with_gpu'] = 0

                            seed = 2025 
                            simu_p = simulation_para.Simu_Para(seed, configs)
                            sig_p = signal_para.Sig_Para(seed, configs)
                            tx_p = txsignal_para.Tx_Para(seed, configs)
                            rx_p = rxsignal_para.Rx_para(seed, configs)
                            ch_p = channel_para.Ch_Para(seed, configs)
                            plot_p = sigplot_para.Plot_Para(seed, configs)
                            
                            simu_p.fig_plot = False
                            plot_p.fig_plot = False 
                            
                            bit, sym, sym_map, integer = sig_main.sig_tx(sig_p, seed=seed)
                            if not isinstance(sym, list): sym = [sym, sym]
                            elif len(sym) == 1: sym = [sym[0], sym[0]]
                            
                            tx_signal = tx_main.tx(sym, tx_p, plot_p) 
                            
                            ch_input = [tx_signal] if isinstance(tx_signal, list) and len(tx_signal) == 2 else tx_signal
                            rx_signal_opt = channel_main.channel_transmission(ch_input, ch_p, plot_para=plot_p)
                            
                            rx_input = rx_signal_opt[0] if isinstance(rx_signal_opt, list) and len(rx_signal_opt) == 1 else rx_signal_opt
                            if isinstance(rx_input, list) and len(rx_input) < 2: rx_input = [rx_input[0], rx_input[0]] 

                            rx_signal_dsp = rx_main.rx(rx_input, sym_map, sym, rx_p, plot_p)
                            
                            st.success(f"✅ 深度仿真完成！(距离: {params['distance']}km, 功率: {params['tx_power']}dBm)")
                            
                            data_x = rx_signal_dsp[0] if isinstance(rx_signal_dsp, list) else rx_signal_dsp
                                
                            def to_numpy(d):
                                if hasattr(d, 'detach'): return d.detach().cpu().numpy()
                                if hasattr(d, 'cpu'): return d.cpu().numpy()
                                if isinstance(d, np.ndarray): return d
                                return np.array(d)
                            
                            data_x = to_numpy(data_x)
                            plot_len = min(2000, len(data_x))
                            data_view = data_x[:plot_len]

                            fig_sim, ax_sim = plt.subplots(1, 2, figsize=(10, 4))
                            ax_sim[0].scatter(data_view.real, data_view.imag, s=5, c='#00FFFF', alpha=0.6, edgecolors='none')
                            ax_sim[0].set_title(f"Rx Constellation ({params['modulation']})")
                            ax_sim[0].grid(True, alpha=0.3)
                            ax_sim[0].set_aspect('equal')
                            max_val = np.max(np.abs(data_view)) * 1.2
                            if max_val < 0.1 or np.isnan(max_val): max_val = 1.0
                            ax_sim[0].set_xlim(-max_val, max_val)
                            ax_sim[0].set_ylim(-max_val, max_val)

                            phases = np.angle(data_view)
                            ax_sim[1].hist(phases, bins=50, color='#FF4B4B', alpha=0.8)
                            ax_sim[1].set_title("Phase Distribution")
                            ax_sim[1].set_ylabel("Count")
                            
                            st.pyplot(fig_sim)
                            st.caption("注：该结果由 PyTorch 神经网络与分步傅里叶方法实时计算生成。")
                    except Exception as e:
                        st.error(f"仿真运行出错: {str(e)}")

@st.fragment
def render_tab5_device_test(params):
    st.header("🛠️ 光无源器件特性测试 (高保真模式)")
    st.markdown("""
    > **⚠️ 物理模型说明**：
    > 此模式已启用 **工程级误差模型**，包含以下真实效应：
    > * **IL**: 器件固有插入损耗。
    > * **Connector Loss**: 可调接头基础损耗、污染/松动附加损耗。
    > * **Sensitivity Limit**: 光功率计底噪 (-70 dBm) 与饱和。
    > * **Measurement Noise**: 可开关的 OPM 测量抖动；关闭后数值按理论模型稳定显示。
    """)
    CONN_LOSS = 0.25      
    NOISE_FLOOR = -70.0   
    SATURATION_P = 10.0   
    sys_wavelength_str = params['wavelength'] 
    sys_lam = 1550 if "1550" in sys_wavelength_str else 1310

    with st.expander("光衰减/光纤损耗 与 OPM 测试 (含多点扫参)", expanded=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            test_mode = st.radio("测试模式", ["单点 VOA + 光纤测试", "多点光纤长度扫参 (OptiSystem 模式)"])
            voa_input_p = st.number_input("光源输入功率 (dBm)", value=0.0, step=0.5, key="voa_in")
            if test_mode == "单点 VOA + 光纤测试":
                voa_att = st.slider("VOA 设置衰减量 (dB)", 0.0, 60.0, 10.0, 0.5, key="voa_set")
                fiber_type = st.selectbox("串联光纤类型", ["无 (直连光功率计)", "G.652 标准单模光纤 (SMF)", "G.655 非零色散位移光纤 (NZDSF)", "OM3 多模光纤 (MMF)"])
                if fiber_type != "无 (直连光功率计)":
                    spool_len = st.number_input("光纤盘长度 (km)", min_value=0.1, max_value=200.0, value=20.0, step=1.0)
                    is_1550 = "1550" in params['wavelength']
                    if "G.652" in fiber_type: alpha = 0.20 if is_1550 else 0.35
                    elif "G.655" in fiber_type: alpha = 0.22 if is_1550 else 0.40 
                    elif "OM3" in fiber_type:
                        alpha = 2.0 if is_1550 else 0.6  
                        if is_1550: st.warning("⚠️ 多模光纤通常不用于 1550nm 传输，本征损耗极大！")
                    st.caption(f"当前波长 {params['wavelength'][:6]} 下，参考衰减系数: {alpha} dB/km")
                else:
                    spool_len = 0.0
                    alpha = 0.0

                # 【新增】光衰减器/跳线接头影响：默认值与原模型一致，可模拟污染、松动与重复插拔误差。
                default_conn_count = 3 if fiber_type != "无 (直连光功率计)" else 2
                with st.expander("🔌 光衰减器接头影响", expanded=False):
                    conn_count_single = st.number_input("接头数量", min_value=0, max_value=12, value=default_conn_count, step=1, key="voa_conn_count")
                    conn_loss_single = st.number_input("单个接头损耗 (dB/个)", min_value=0.00, max_value=2.00, value=CONN_LOSS, step=0.05, key="voa_conn_loss")
                    dirty_loss_single = st.number_input("接头污染/松动附加损耗 (dB/个)", min_value=0.00, max_value=5.00, value=0.00, step=0.05, key="voa_dirty_loss")
                    enable_noise_single = st.checkbox("启用随机测量误差", value=False, key="voa_enable_noise", help="关闭时 OPM 读数按理论值稳定显示；开启后模拟重复插拔和光功率计读数抖动。")
                    repeat_std_single = st.number_input("重复插拔测量抖动 σ (dB)", min_value=0.00, max_value=1.00, value=0.03, step=0.01, key="voa_repeat_std", disabled=not enable_noise_single)
                    return_loss_db_single = st.slider("回波损耗 RL (dB，仅用于风险提示)", 20.0, 65.0, 45.0, 1.0, key="voa_return_loss")
                    st.caption("接头损耗 = 接头数量 × (单个接头损耗 + 污染/松动附加损耗)。接头数量设为 0 时，接头相关损耗严格为 0。")
            else:
                lengths_str = st.text_input("测试长度 (km，可输入单个数或逗号分隔列表)", "2, 10, 23, 29, 35")
                st.caption(f"当前光纤衰减系数: {params['attenuation']} dB/km (读取自侧边栏)。单个长度用于输出一次 OPM 数值；多个长度用于线性拟合衰减系数。")
                optisystem_align = st.checkbox("OptiSystem 理想对齐模式", value=True, key="optisystem_align_sweep", help="开启后按 CW Laser → Optical Fiber → Optical Power Meter 的理想模型计算，默认不加入连接器、VOA 插损和随机测量抖动。")
                if optisystem_align:
                    fixed_device_loss_user = 0.0
                    conn_count_user = 0
                    conn_loss_sweep_user = 0.0
                    dirty_loss_sweep_user = 0.0
                    measurement_std_user = 0.0
                else:
                    fixed_device_loss_user = st.number_input("固定器件插损 (dB)", min_value=0.0, max_value=10.0, value=0.80, step=0.05, key="sweep_fixed_loss")
                    conn_count_user = st.number_input("连接器数量", min_value=0, max_value=10, value=2, step=1, key="sweep_conn_count")
                    conn_loss_sweep_user = st.number_input("单个接头损耗 (dB/个)", min_value=0.00, max_value=2.00, value=CONN_LOSS, step=0.05, key="sweep_conn_loss")
                    dirty_loss_sweep_user = st.number_input("接头污染/松动附加损耗 (dB/个)", min_value=0.00, max_value=5.00, value=0.00, step=0.05, key="sweep_dirty_loss")
                    enable_noise_sweep = st.checkbox("启用 OPM 随机测量误差", value=False, key="sweep_enable_noise", help="关闭时多点扫参不加入随机扰动，便于和理论值/OptiSystem 对齐。")
                    measurement_std_user = st.number_input("OPM测量抖动 σ (dB)", min_value=0.0, max_value=1.0, value=0.03, step=0.01, key="sweep_meas_std", disabled=not enable_noise_sweep)
                voa_att = 0.0 
                
        with c2:
            if test_mode == "单点 VOA + 光纤测试":
                voa_insertion_loss = 0.8 
                if fiber_type != "无 (直连光功率计)":
                    fiber_loss = alpha * spool_len
                else:
                    fiber_loss = 0.0

                conn_count = int(conn_count_single)
                connector_base_loss = conn_count * float(conn_loss_single)
                connector_dirty_loss = conn_count * float(dirty_loss_single)
                connector_total_loss = connector_base_loss + connector_dirty_loss
                total_loss = voa_att + voa_insertion_loss + connector_total_loss + fiber_loss
                power_at_detector = voa_input_p - total_loss
                measurement_std_single = float(repeat_std_single) if enable_noise_single else 0.0
                measurement_error = np.random.normal(0, measurement_std_single) if measurement_std_single > 0 else 0.0
                measured_power = power_at_detector + measurement_error
                
                if power_at_detector > SATURATION_P:
                    display_text, display_color = "HI (Overload)", "#FF0000"
                elif power_at_detector < NOISE_FLOOR:
                    floor_reading = NOISE_FLOOR + (np.random.normal(0, 1.5) if measurement_std_single > 0 else 0.0)
                    display_text, display_color = f"{floor_reading:.2f} dBm", "#555555"
                else:
                    display_text, display_color = f"{measured_power:.2f} dBm", "#00FF00" 

                st.markdown(f"""
                <div class="device-box" style="border-color: {display_color};">
                    <div class="lcd-display" style="color: {display_color}; font-size: 2.2rem;">{display_text}</div>
                    <p style="margin:0; font-weight:bold; color:#ddd;">光功率计读数 (OPM)</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                **📉 链路损耗清单（总计 {total_loss:.2f} dB）：**
                * VOA 设定衰减：{voa_att:.2f} dB
                * VOA 固有插入损耗：{voa_insertion_loss:.2f} dB
                * 光纤盘损耗：{fiber_loss:.2f} dB
                * 接头基础损耗：{conn_count} × {float(conn_loss_single):.2f} = {connector_base_loss:.2f} dB
                * 接头污染/松动损耗：{conn_count} × {float(dirty_loss_single):.2f} = {connector_dirty_loss:.2f} dB
                * OPM 测量随机误差：{'开启' if measurement_std_single > 0 else '关闭'}，σ = {measurement_std_single:.2f} dB
                * 理论输出功率：{power_at_detector:.2f} dBm
                """)
                if return_loss_db_single < 30:
                    st.warning("⚠️ 当前回波损耗较低，说明反射较强；真实实验中可能引起激光器输出波动或读数不稳定。")
                elif conn_count == 0:
                    st.info("当前接头数量为 0，接头基础损耗与污染损耗均为 0；读数只受 VOA、固有插损、光纤损耗和测量误差影响。")
            else:
                # 多点光纤长度扫参：恢复并增强原有数据表与拟合曲线输出。
                # 该模式模拟光源、两端连接器、固定器件插损和光功率计测量误差，
                # 用多点线性拟合反推出光纤衰减系数，更接近真实 OPM/光纤盘测试流程。
                try:
                    lengths = [float(x.strip()) for x in lengths_str.replace("，", ",").split(",") if x.strip()]
                    lengths = [x for x in lengths if x >= 0]
                except Exception:
                    lengths = []

                if len(lengths) == 0:
                    st.warning("请输入有效长度，例如：20 或 2, 10, 23, 29, 35")
                else:
                    alpha_true = float(params['attenuation'])
                    fixed_device_loss = float(fixed_device_loss_user)
                    conn_count = int(conn_count_user)
                    connector_base_loss = conn_count * float(conn_loss_sweep_user)
                    connector_dirty_loss = conn_count * float(dirty_loss_sweep_user)
                    connector_total_loss = connector_base_loss + connector_dirty_loss
                    fixed_loss = fixed_device_loss + connector_total_loss
                    measurement_std = float(measurement_std_user) if (not optisystem_align and 'enable_noise_sweep' in locals() and enable_noise_sweep) else 0.0

                    rows = []
                    for length_km in lengths:
                        fiber_loss = alpha_true * length_km
                        ideal_power = voa_input_p - fixed_loss - fiber_loss
                        if ideal_power > SATURATION_P:
                            measured_power = SATURATION_P
                            status = "过载"
                        elif ideal_power < NOISE_FLOOR:
                            measured_power = NOISE_FLOOR + (np.random.normal(0, 1.5) if measurement_std > 0 else 0.0)
                            status = "低于底噪"
                        else:
                            measured_power = ideal_power + (np.random.normal(0, measurement_std) if measurement_std > 0 else 0.0)
                            status = "有效"

                        rows.append({
                            "光纤长度 (km)": length_km,
                            "理论光纤损耗 (dB)": fiber_loss,
                            "固定器件插损 (dB)": fixed_device_loss,
                            "接头基础损耗 (dB)": connector_base_loss,
                            "接头污染损耗 (dB)": connector_dirty_loss,
                            "总固定损耗 (dB)": fixed_loss,
                            "理论接收功率 (dBm)": ideal_power,
                            "OPM读数 (dBm)": measured_power,
                            "测量状态": status,
                        })

                    df_sweep = pd.DataFrame(rows)
                    valid_df = df_sweep[df_sweep["测量状态"] == "有效"].copy()

                    if len(lengths) == 1:
                        st.markdown("#### 单点光纤长度测量结果")
                        row0 = df_sweep.iloc[0]
                        c_one1, c_one2, c_one3 = st.columns(3)
                        c_one1.metric("光纤长度", f"{row0['光纤长度 (km)']:.2f} km")
                        c_one2.metric("理论光纤损耗", f"{row0['理论光纤损耗 (dB)']:.2f} dB")
                        c_one3.metric("OPM读数", f"{row0['OPM读数 (dBm)']:.2f} dBm")
                        st.caption("单点长度只能得到该长度下的接收功率；若要反推出衰减系数，请输入两个及以上长度点。")
                    else:
                        st.markdown("#### 多点光纤长度扫参结果")

                    st.dataframe(df_sweep.style.format({
                        "光纤长度 (km)": "{:.2f}",
                        "理论光纤损耗 (dB)": "{:.2f}",
                        "固定器件插损 (dB)": "{:.2f}",
                        "接头基础损耗 (dB)": "{:.2f}",
                        "接头污染损耗 (dB)": "{:.2f}",
                        "总固定损耗 (dB)": "{:.2f}",
                        "理论接收功率 (dBm)": "{:.2f}",
                        "OPM读数 (dBm)": "{:.2f}",
                    }), use_container_width=True)

                    fig_len, ax_len = plt.subplots(figsize=(7, 3.2))
                    ax_len.scatter(df_sweep["光纤长度 (km)"], df_sweep["OPM读数 (dBm)"], s=55, label="OPM测量点")
                    ax_len.plot(df_sweep["光纤长度 (km)"], df_sweep["理论接收功率 (dBm)"], linestyle="--", linewidth=1.8, label="理论线性衰减")

                    if len(valid_df) >= 2:
                        fit_k, fit_b = np.polyfit(valid_df["光纤长度 (km)"], valid_df["OPM读数 (dBm)"], 1)
                        fit_x = np.linspace(min(lengths), max(lengths), 100)
                        fit_y = fit_k * fit_x + fit_b
                        estimated_alpha = -fit_k
                        ax_len.plot(fit_x, fit_y, linewidth=2.0, label=f"线性拟合 α≈{estimated_alpha:.3f} dB/km")

                        c_fit1, c_fit2, c_fit3 = st.columns(3)
                        c_fit1.metric("拟合衰减系数", f"{estimated_alpha:.3f} dB/km")
                        c_fit2.metric("侧边栏设定值", f"{alpha_true:.3f} dB/km")
                        c_fit3.metric("估计误差", f"{abs(estimated_alpha-alpha_true):.3f} dB/km")
                    elif len(lengths) > 1:
                        st.info("有效测量点不足，无法拟合衰减系数。请降低输入功率避免过载，或缩短长度避免低于底噪。")

                    ax_len.axhline(NOISE_FLOOR, linestyle=":", alpha=0.7, label="OPM底噪")
                    ax_len.axhline(SATURATION_P, linestyle=":", alpha=0.7, label="OPM饱和")
                    ax_len.set_xlabel("光纤长度 (km)")
                    ax_len.set_ylabel("接收光功率 (dBm)")
                    ax_len.set_title("单点/多点长度法测量光纤功率衰减")
                    if len(lengths) == 1:
                        x0 = lengths[0]
                        ax_len.set_xlim(max(-0.5, x0 - 5), x0 + 5)
                    ax_len.grid(True, alpha=0.25)
                    ax_len.legend(loc="best")
                    st.pyplot(fig_len)

                    if optisystem_align:
                        st.caption("说明：当前为 OptiSystem 理想对齐模式，按 P_rx = P_tx - αL 计算；适合对齐 CW Laser → Optical Fiber → Optical Power Meter 的基础扫参结果。")
                    else:
                        st.caption(f"说明：真实实验中通常记录不同光纤长度下的 OPM 功率读数，并用功率-长度斜率估计衰减系数；本模型已拆分固定器件插损、接头基础损耗、接头污染/松动损耗、底噪、饱和和测量抖动。当前测量随机误差：{'开启' if measurement_std > 0 else '关闭'}，σ = {measurement_std:.2f} dB。")

    with st.expander("光隔离器 (ISO) - 波长敏感性测试"):
        c1, c2, c3 = st.columns(3)
        with c1:
            iso_spec = st.selectbox("器件规格", ["1550 nm (C-Band)", "1310 nm (O-Band)"])
            iso_iso_rating = st.slider("标称隔离度 (dB)", 20.0, 60.0, 40.0, 5.0)
        with c2:
            direction = st.radio("光传输方向", ["➡️ 正向 (Tx -> Rx)", "⬅️ 反向 (反射光)"])
            input_p_iso = st.number_input("输入功率 (dBm)", value=0.0, key="iso_in")
        with c3:
            iso_il_base = 0.6 
            iso_center = 1550 if "1550" in iso_spec else 1310
            wl_diff = abs(sys_lam - iso_center)
            
            isolation_penalty_factor = 1.0
            if wl_diff > 100: 
                isolation_penalty_factor = 0.0 
                st.error("❌ 波长严重失配！隔离器失效！")
            
            real_isolation = max(0.5, iso_iso_rating * isolation_penalty_factor)
            
            if "正向" in direction:
                loss = iso_il_base + (2 * CONN_LOSS)
                if wl_diff > 100: loss += 2.0 
                out_p_iso = input_p_iso - loss
                loss_type = "插入损耗 (IL)"
            else:
                loss = real_isolation + (2 * CONN_LOSS)
                out_p_iso = input_p_iso - loss
                loss_type = "隔离损耗 (Iso)"

            st.metric("输出功率", f"{out_p_iso:.2f} dBm")
            st.caption(f"📉 当前经历: {loss_type} = {loss:.2f} dB")

    with st.expander("熔融拉锥分路器 (Coupler) - 附加损耗"):
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            split_ratio = st.slider("分光比 (Port 1 %)", 0, 100, 50, 1)
            splitter_in = st.number_input("输入功率 (dBm)", value=0.0, key="split_in")
            excess_loss = 0.15 
        with col_s2:
            ratio_1 = split_ratio / 100.0
            ratio_2 = 1.0 - ratio_1
            loss_split_1 = -10 * math.log10(ratio_1) if ratio_1 > 0 else 60
            loss_split_2 = -10 * math.log10(ratio_2) if ratio_2 > 0 else 60
            
            p1 = splitter_in - loss_split_1 - excess_loss - CONN_LOSS
            p2 = splitter_in - loss_split_2 - excess_loss - CONN_LOSS
            
            fig_split, ax_split = plt.subplots(figsize=(6, 2.5))
            y_pos = [0, 1]
            powers = [p2, p1]
            labels = [f"Port 2 ({int(ratio_2*100)}%)", f"Port 1 ({int(ratio_1*100)}%)"]
            bars = ax_split.barh(y_pos, powers, color=['#FF4B4B', '#00FFFF'], height=0.6)
            ax_split.set_yticks(y_pos)
            ax_split.set_yticklabels(labels)
            ax_split.set_xlabel("Output Power (dBm)")
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width if width > -60 else -60
                ax_split.text(label_x_pos + 1, bar.get_y() + bar.get_height()/2, f'{powers[i]:.2f} dBm', va='center', color='white', fontweight='bold')
            ax_split.set_xlim(splitter_in - 20, splitter_in + 5)
            st.pyplot(fig_split)

    with st.expander("光波分复用器 (WDM) - 谱分析"):
        w1, w2 = st.columns(2)
        with w1:
            p_1310 = st.slider("1310nm 信道功率 (dBm)", -30.0, 10.0, -5.0)
            enable_1310 = st.checkbox("启用 1310nm", value=True)
        with w2:
            p_1550 = st.slider("1550nm 信道功率 (dBm)", -30.0, 10.0, -2.0)
            enable_1550 = st.checkbox("启用 1550nm", value=True)
            
        x_lambda = np.linspace(1250, 1600, 500)
        y_spectrum = np.ones_like(x_lambda) * -60 
        def add_peak(x, center, power, width=10): return power * np.exp(-((x - center)**2) / (2 * width**2))
        
        if enable_1310:
            peak = add_peak(x_lambda, 1310, 10**(p_1310/10), width=5)
            y_spectrum = np.maximum(y_spectrum, 10*np.log10(peak + 1e-9))
        if enable_1550:
            peak = add_peak(x_lambda, 1550, 10**(p_1550/10), width=5)
            y_spectrum = np.maximum(y_spectrum, 10*np.log10(peak + 1e-9))
            
        fig_wdm, ax_wdm = plt.subplots(figsize=(8, 3))
        ax_wdm.plot(x_lambda, y_spectrum, color='#FFFF00', linewidth=1.5)
        ax_wdm.fill_between(x_lambda, -70, y_spectrum, color='#FFFF00', alpha=0.2)
        ax_wdm.set_ylim(-70, 15)
        ax_wdm.set_xlabel("波长 (nm)")
        ax_wdm.set_ylabel("PSD (dBm)")
        ax_wdm.axvline(1310, color='cyan', linestyle=':', alpha=0.5)
        ax_wdm.text(1310, 10, "O-Band", color='cyan', ha='center')
        ax_wdm.axvline(1550, color='red', linestyle=':', alpha=0.5)
        ax_wdm.text(1550, 10, "C-Band", color='red', ha='center')
        st.pyplot(fig_wdm)

@st.fragment
def render_tab_ld():
    st.header("💡 半导体 LD 光源 P-I 与直接调制特性")
    st.markdown("探究注入电流、温度、阈值电流、斜率效率和直接调制响应的关系。该模块采用工程近似模型，用于贴近真实 LD 台架实验趋势。")
    col_ld1, col_ld2 = st.columns([1, 2])
    with col_ld1:
        st.subheader("参数设置")
        ld_temp = st.slider("工作温度 (°C)", 10.0, 60.0, 25.0, 1.0)
        ld_current = st.slider("当前注入电流 (mA)", 0.0, 80.0, 12.0, 0.1)
        ld_bias_voltage = st.slider("LD 偏置电压 Vbias (V)", 0.0, 3.0, 1.8, 0.01)
        ld_series_resistance = st.slider("等效串联电阻 Rs (Ω)", 1.0, 30.0, 8.0, 0.5)
        voltage_control_enabled = st.checkbox("用偏置电压联动注入电流", value=False, help="关闭时保持原来的电流控制；开启后根据 Vbias、导通电压和 Rs 估算注入电流。")
        chip_slope_efficiency = st.slider("芯片斜率效率 (mW/mA)", 0.05, 0.60, 0.20, 0.01)
        coupling_loss_db = st.slider("耦合损耗 (dB)", 1.0, 8.0, 4.0, 0.5)
        connector_loss_db = 0.25

        base_ith = 6.0
        T0 = 40.0
        current_ith = base_ith * np.exp((ld_temp - 25.0) / T0)
        temp_slope_penalty = np.exp(-(ld_temp - 25.0) / 120.0)
        effective_slope = chip_slope_efficiency * temp_slope_penalty
        turn_on_voltage = 1.18 + 0.002 * (ld_temp - 25.0)
        voltage_est_current = max((ld_bias_voltage - turn_on_voltage) / max(ld_series_resistance, 1e-9) * 1000.0, 0.0)
        drive_current_for_model = voltage_est_current if voltage_control_enabled else ld_current
        thermal_voltage_penalty_db = max(0.0, (ld_bias_voltage - 2.4) * 2.0)

        if drive_current_for_model < current_ith:
            out_power_mw_chip = 0.0008 * drive_current_for_model
            state = "自发辐射/LED 区"
        else:
            out_power_mw_chip = 0.0008 * current_ith + effective_slope * (drive_current_for_model - current_ith)
            state = "受激辐射/Laser 区"

        out_power_dbm = 10 * np.log10(max(out_power_mw_chip, 1e-12)) - coupling_loss_db - connector_loss_db - thermal_voltage_penalty_db
        out_power_dbm = max(out_power_dbm, -70.0)
        real_output_mw = 10 ** (out_power_dbm / 10.0)

        bias_ratio = max((drive_current_for_model - current_ith) / max(current_ith, 1e-9), 0.0)
        relaxation_freq_ghz = 0.0 if bias_ratio <= 0 else 1.8 * np.sqrt(bias_ratio)
        rin_db_hz = -145 + 18 * np.exp(-bias_ratio) + max(0, (ld_temp - 25) * 0.15)

        st.metric("阈值电流 Ith", f"{current_ith:.2f} mA")
        st.metric("等效驱动电流", f"{drive_current_for_model:.2f} mA")
        st.metric("偏置电压/导通电压", f"{ld_bias_voltage:.2f} / {turn_on_voltage:.2f} V")
        st.metric("终端输出光功率", f"{out_power_dbm:.2f} dBm ({real_output_mw:.3f} mW)")
        st.metric("驰张振荡频率 fr", f"{relaxation_freq_ghz:.2f} GHz")
        st.metric("相对强度噪声 RIN", f"{rin_db_hz:.1f} dB/Hz")
        st.info(f"当前工作状态: {state}")

    with col_ld2:
        i_array = np.linspace(0, 100, 300)
        ith = current_ith
        spontaneous_emission = 0.0008 * np.minimum(i_array, ith)
        stimulated_emission = np.where(i_array < ith, 0.0, effective_slope * (i_array - ith))
        raw_p_array = spontaneous_emission + stimulated_emission
        p_array_dbm = 10 * np.log10(np.maximum(raw_p_array, 1e-12)) - coupling_loss_db - connector_loss_db
        p_array_dbm = np.maximum(p_array_dbm, -70.0)
        terminal_p_array = 10 ** (p_array_dbm / 10.0)

        fig_ld, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(i_array, terminal_p_array, linewidth=2, label='线性 P-I (mW)')
        ax1.scatter([drive_current_for_model], [real_output_mw], s=70, zorder=5, label='当前工作点')
        ax1.axvline(current_ith, linestyle='--', label=f'阈值电流 ({current_ith:.1f} mA)')
        ax1.set_xlabel("注入电流 (mA)")
        ax1.set_ylabel("终端输出光功率 (mW)")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(i_array, p_array_dbm, linestyle=':', linewidth=2, alpha=0.75, label='对数 P-I (dBm)')
        ax2.set_ylabel("终端输出光功率 (dBm)")
        ax2.set_ylim(-40, 10)
        plt.title("LD 光源 P-I 特性曲线（温度与耦合损耗修正）")
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
        st.pyplot(fig_ld)

    st.markdown("---")
    with st.expander("🔬 LD 直接调制动态响应：驰张振荡与自脉动趋势", expanded=False):
        c_dyn1, c_dyn2 = st.columns([1, 2])
        with c_dyn1:
            mod_freq_ghz = st.slider("调制频率 (GHz)", 0.1, 20.0, 2.5, 0.1)
            mod_depth = st.slider("调制深度", 0.05, 1.00, 0.35, 0.05)
            damping_ghz = st.slider("阻尼因子 γ (GHz)", 0.5, 12.0, 4.0, 0.5)
            alpha_h = st.slider("线宽增强因子 αH", 1.0, 8.0, 4.0, 0.5)
            enable_ld_noise = st.checkbox("叠加 RIN 噪声", value=True, key="enable_ld_rin_noise")
            ld_noise_scale = st.slider("LD 噪声强度系数", 0.0, 5.0, 1.0, 0.1, key="ld_noise_scale")

        f = np.linspace(0.01, 20.0, 500)
        fr = max(relaxation_freq_ghz, 0.05)
        response = (fr ** 2) / np.sqrt((fr ** 2 - f ** 2) ** 2 + (damping_ghz * f) ** 2)
        response_db = 20 * np.log10(response / max(response[0], 1e-12))

        t_dyn = np.linspace(0, 5, 1200)  # ns
        drive = np.sin(2 * np.pi * mod_freq_ghz * t_dyn)
        resonance_gain = np.interp(mod_freq_ghz, f, response / max(response[0], 1e-12))
        self_pulsing_flag = (bias_ratio < 0.35 and mod_depth > 0.55) or (relaxation_freq_ghz > 0 and abs(mod_freq_ghz - relaxation_freq_ghz) < 0.25 * relaxation_freq_ghz and mod_depth > 0.45)
        sp_component = 0.0
        if self_pulsing_flag:
            sp_component = 0.22 * np.sin(2 * np.pi * max(fr, 0.5) * t_dyn) * np.exp(-t_dyn / 4)
        p_norm = 1.0 + mod_depth * resonance_gain * drive + sp_component
        if enable_ld_noise and ld_noise_scale > 0:
            rng_ld = np.random.default_rng(2025)
            rin_linear = 10 ** (rin_db_hz / 10.0)
            noise_sigma = min(0.25, np.sqrt(max(rin_linear * 12.5e9, 0.0)) * 0.02 * ld_noise_scale)
            p_norm = p_norm + rng_ld.normal(0.0, noise_sigma, len(p_norm))
        p_norm = np.clip(p_norm, 0.02, None)
        chirp = alpha_h * np.gradient(np.log(p_norm), t_dyn)

        with c_dyn2:
            fig_resp, ax_resp = plt.subplots(figsize=(8, 3.5))
            ax_resp.plot(f, response_db, linewidth=2)
            ax_resp.axvline(fr, linestyle='--', alpha=0.7, label=f'fr={fr:.2f} GHz')
            ax_resp.axvline(mod_freq_ghz, linestyle=':', alpha=0.7, label=f'调制频率={mod_freq_ghz:.1f} GHz')
            ax_resp.set_xlabel("调制频率 (GHz)")
            ax_resp.set_ylabel("归一化响应 (dB)")
            ax_resp.set_title("LD 小信号调制响应曲线")
            ax_resp.legend()
            st.pyplot(fig_resp)

            fig_time, ax_time = plt.subplots(figsize=(8, 3.5))
            ax_time.plot(t_dyn, p_norm, linewidth=1.6, label='归一化输出光功率')
            ax_time.plot(t_dyn, chirp / max(np.max(np.abs(chirp)), 1e-9) * 0.4 + 1.0, linestyle=':', alpha=0.8, label='归一化啁啾趋势')
            ax_time.set_xlabel("时间 (ns)")
            ax_time.set_ylabel("归一化幅度")
            ax_time.set_title("直接调制时域输出与啁啾趋势")
            ax_time.legend()
            st.pyplot(fig_time)

        if self_pulsing_flag:
            st.warning("⚠️ 当前偏置/调制条件下可能出现明显驰张振荡增强或自脉动趋势，真实实验中应降低调制深度或提高偏置电流。")
        else:
            st.success("✅ 当前直接调制条件相对稳定，未触发明显自脉动判据。")

@st.fragment
def render_tab_fiber_param():
    st.header("🔀 光纤截止波长与模式分析")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        core_radius = st.number_input("纤芯半径 a (μm)", min_value=1.0, max_value=50.0, value=4.15, step=0.1)
        n1 = st.number_input("纤芯折射率 n1", min_value=1.4, max_value=1.6, value=1.468, step=0.001)
        n2 = st.number_input("包层折射率 n2", min_value=1.4, max_value=1.6, value=1.463, step=0.001)
        test_lambda = st.slider("测试波长 (nm)", 800.0, 1700.0, 1550.0, 10.0)
        
        NA = np.sqrt(n1**2 - n2**2) if n1 > n2 else 0
        V_param = (2 * np.pi * core_radius * 1e-6 / (test_lambda * 1e-9)) * NA
        cutoff_lambda_nm = (2 * np.pi * core_radius * 1e-6 / 2.405) * NA * 1e9
        
    with col_f2:
        st.markdown("#### 📊 核心计算结果")
        st.metric("数值孔径 (NA)", f"{NA:.4f}")
        st.metric("理论截止波长 (λc)", f"{cutoff_lambda_nm:.1f} nm")
        st.metric(f"@{test_lambda}nm 归一化频率 (V)", f"{V_param:.3f}")
        
        if V_param < 2.405 and V_param > 0:
            st.success("✅ **单模传输状态**：当前波长大于截止波长，光纤内仅存在基模 (LP01)。")
        elif V_param >= 2.405:
            mode_count = int(V_param**2 / 2)
            st.warning(f"⚠️ **多模传输状态**：当前存在约 {mode_count} 个传播模式，会产生严重的模式色散。")
        else:
            st.error("❌ 折射率设置错误 (n1 必须大于 n2)")

@st.fragment

@st.fragment
def render_tab_fiber_principle():
    st.header("🧭 光纤传输原理实验：几何光学方法与波动理论")
    st.markdown("""
    本实验用于解释光纤为什么能够导光、为什么会出现单模/多模，以及截止波长与 V 参数的关系。
    模型采用教学型近似，适合与后续的损耗、带宽、截止波长、色散和系统设计实验衔接。
    """)

    sub_geo, sub_wave, sub_compare = st.tabs(["① 几何光学导光", "② 波动理论与模式", "③ 对比分析与报告"])

    with sub_geo:
        st.subheader("① 几何光学方法：全反射、数值孔径与光线轨迹")
        col_ctrl, col_plot = st.columns([1, 1.45])
        with col_ctrl:
            st.markdown("#### 参数设置")
            n0 = st.number_input("外界介质折射率 n₀", min_value=1.000, max_value=1.600, value=1.000, step=0.001, format="%.3f", key="fp_n0")
            n1 = st.number_input("纤芯折射率 n₁", min_value=1.300, max_value=1.700, value=1.468, step=0.001, format="%.3f", key="fp_n1")
            n2 = st.number_input("包层折射率 n₂", min_value=1.250, max_value=1.690, value=1.462, step=0.001, format="%.3f", key="fp_n2")
            core_radius_um = st.slider("纤芯半径 a (μm)", 3.0, 62.5, 25.0, 0.5, key="fp_core_radius")
            theta_in_deg = st.slider("空气中入射角 θ₀ (°)", 0.0, 45.0, 8.0, 0.5, key="fp_theta_in")
            alpha_geo = st.slider("光纤衰减系数 α (dB/km)", 0.15, 0.50, 0.20, 0.01, key="fp_alpha_geo")
            length_geo = st.slider("传输长度 L (km)", 1.0, 50.0, 10.0, 1.0, key="fp_length_geo")

            if n1 <= n2:
                st.error("参数不合理：纤芯折射率 n₁ 必须大于包层折射率 n₂。")
                return
            na = float(np.sqrt(max(n1**2 - n2**2, 0.0)))
            theta_accept_deg = float(np.degrees(np.arcsin(min(na / max(n0, 1e-12), 1.0))))
            theta_c_deg = float(np.degrees(np.arcsin(min(n2 / n1, 1.0))))
            sin_theta_core = min(n0 * np.sin(np.radians(theta_in_deg)) / n1, 1.0)
            theta_core_deg = float(np.degrees(np.arcsin(sin_theta_core)))
            guided = theta_in_deg <= theta_accept_deg + 1e-9
            pout_ratio = 10 ** (-alpha_geo * length_geo / 10.0) if guided else 0.0

            st.markdown("#### 计算结果")
            g1, g2 = st.columns(2)
            g1.metric("数值孔径 NA", f"{na:.4f}")
            g2.metric("最大接收角 θmax", f"{theta_accept_deg:.2f}°")
            g3, g4 = st.columns(2)
            g3.metric("临界角 θc", f"{theta_c_deg:.2f}°")
            g4.metric("输出功率比例", f"{pout_ratio:.3f}")
            if guided:
                st.success("✅ 当前入射角满足接收条件，光线可在纤芯中形成全反射导光。")
            else:
                st.warning("⚠️ 当前入射角超过最大接收角，耦合效率很低，不能形成有效导光。")

        with col_plot:
            st.markdown("#### 光线传播示意")
            a = core_radius_um
            z_max = 10.0
            z = np.linspace(0, z_max, 1200)
            if guided and theta_core_deg > 0.05:
                slope = np.tan(np.radians(theta_core_deg)) * z_max / 2.8
                raw_y = slope * z
                period = 4 * a
                y = ((raw_y + a) % period)
                y = np.where(y <= 2 * a, y - a, 3 * a - y)
            elif guided:
                y = np.zeros_like(z)
            else:
                y = np.tan(np.radians(max(theta_core_deg, 0.5))) * z * z_max / 2.8
                y = np.clip(y, -a * 1.4, a * 1.4)

            fig, ax = plt.subplots(figsize=(8, 3.7))
            ax.fill_between(z, -a, a, alpha=0.18, label="纤芯")
            ax.axhline(a, linestyle="--", linewidth=1.2, label="纤芯/包层边界")
            ax.axhline(-a, linestyle="--", linewidth=1.2)
            ax.plot(z, y, linewidth=2.2, label="入射光线轨迹")
            if not guided:
                ax.text(z_max * 0.52, a * 0.72, "超过接收角：泄漏/辐射", fontsize=11)
            ax.set_xlabel("轴向传播距离（归一化）")
            ax.set_ylabel("径向位置 (μm)")
            ax.set_title("阶跃型光纤中的几何光学导光示意")
            ax.set_ylim(-a * 1.55, a * 1.55)
            ax.legend(loc="upper right")
            st.pyplot(fig)

            st.caption("说明：该图用于观察全反射与接收角条件，光线路径为教学示意，不代表真实电磁场模式分布。")

    with sub_wave:
        st.subheader("② 波动理论方法：V 参数、单模条件与 LP 模式")
        col_ctrl, col_view = st.columns([1, 1.45])
        with col_ctrl:
            st.markdown("#### 参数设置")
            n1_w = st.number_input("纤芯折射率 n₁ ", min_value=1.300, max_value=1.700, value=1.468, step=0.001, format="%.3f", key="fp_n1_w")
            n2_w = st.number_input("包层折射率 n₂ ", min_value=1.250, max_value=1.690, value=1.462, step=0.001, format="%.3f", key="fp_n2_w")
            radius_w_um = st.slider("纤芯半径 a (μm) ", 2.0, 62.5, 4.5, 0.5, key="fp_radius_w")
            wavelength_nm = st.slider("工作波长 λ (nm)", 800, 1700, 1550, 10, key="fp_wavelength_nm")
            lp_mode = st.selectbox("近似显示的 LP 模式", ["LP01", "LP11", "LP21", "LP02"], key="fp_lp_mode")
            sweep_target = st.radio("参数扫描", ["波长扫描", "纤芯半径扫描"], horizontal=True, key="fp_sweep_target")

            if n1_w <= n2_w:
                st.error("参数不合理：纤芯折射率 n₁ 必须大于包层折射率 n₂。")
                return
            na_w = float(np.sqrt(max(n1_w**2 - n2_w**2, 0.0)))
            wavelength_um = wavelength_nm / 1000.0
            v_number = 2 * np.pi * radius_w_um * na_w / wavelength_um
            mode_count = max(1.0, v_number**2 / 2.0)
            cutoff_nm = 2 * np.pi * radius_w_um * na_w / 2.405 * 1000.0
            is_single_mode = v_number < 2.405

            st.markdown("#### 计算结果")
            w1, w2 = st.columns(2)
            w1.metric("归一化频率 V", f"{v_number:.3f}")
            w2.metric("截止波长 λc", f"{cutoff_nm:.0f} nm")
            w3, w4 = st.columns(2)
            w3.metric("模式数近似 M≈V²/2", f"{mode_count:.1f}")
            w4.metric("光纤状态", "单模" if is_single_mode else "多模")
            if is_single_mode:
                st.success("✅ V < 2.405，当前参数满足阶跃型光纤近似单模条件。")
            else:
                st.info("ℹ️ V ≥ 2.405，当前参数会支持多个传输模式，容易出现模式色散。")

        with col_view:
            st.markdown("#### 模式场强近似分布")
            grid = np.linspace(-1.25, 1.25, 180)
            X, Y = np.meshgrid(grid, grid)
            R = np.sqrt(X**2 + Y**2)
            Phi = np.arctan2(Y, X)
            aperture = R <= 1.0
            if lp_mode == "LP01":
                field = np.exp(-(R / 0.55)**2)
            elif lp_mode == "LP11":
                field = (R / 0.55) * np.cos(Phi) * np.exp(-(R / 0.62)**2)
            elif lp_mode == "LP21":
                field = (R / 0.60)**2 * np.cos(2 * Phi) * np.exp(-(R / 0.70)**2)
            else:
                field = (1 - 2.2 * R**2) * np.exp(-(R / 0.72)**2)
            intensity = np.where(aperture, field**2, np.nan)

            fig_mode, ax_mode = plt.subplots(figsize=(5.0, 4.2))
            im = ax_mode.imshow(intensity, extent=[-1, 1, -1, 1], origin="lower")
            circle = plt.Circle((0, 0), 1, fill=False, linewidth=1.5)
            ax_mode.add_patch(circle)
            ax_mode.set_title(f"{lp_mode} 模式场强分布（归一化示意）")
            ax_mode.set_xlabel("x / a")
            ax_mode.set_ylabel("y / a")
            fig_mode.colorbar(im, ax=ax_mode, fraction=0.046, pad=0.04, label="归一化强度")
            st.pyplot(fig_mode)

            st.markdown("#### V 参数扫描")
            fig_sweep, ax_sweep = plt.subplots(figsize=(7.2, 3.4))
            if sweep_target == "波长扫描":
                lam_arr_nm = np.linspace(800, 1700, 240)
                v_arr = 2 * np.pi * radius_w_um * na_w / (lam_arr_nm / 1000.0)
                ax_sweep.plot(lam_arr_nm, v_arr, linewidth=2)
                ax_sweep.axvline(wavelength_nm, linestyle=":", linewidth=1.2, label="当前 λ")
                ax_sweep.set_xlabel("工作波长 λ (nm)")
            else:
                r_arr_um = np.linspace(2.0, 62.5, 240)
                v_arr = 2 * np.pi * r_arr_um * na_w / wavelength_um
                ax_sweep.plot(r_arr_um, v_arr, linewidth=2)
                ax_sweep.axvline(radius_w_um, linestyle=":", linewidth=1.2, label="当前 a")
                ax_sweep.set_xlabel("纤芯半径 a (μm)")
            ax_sweep.axhline(2.405, linestyle="--", linewidth=1.2, label="单模截止 V=2.405")
            ax_sweep.set_ylabel("V 参数")
            ax_sweep.set_title("V 参数与单模/多模状态变化")
            ax_sweep.legend(loc="best")
            st.pyplot(fig_sweep)

    with sub_compare:
        st.subheader("③ 几何光学与波动理论对比分析")
        st.markdown("""
        | 对比项 | 几何光学方法 | 波动理论方法 |
        |---|---|---|
        | 主要解释对象 | 全反射、临界角、数值孔径、接收角 | V 参数、LP 模式、截止波长、单模/多模 |
        | 适用场景 | 多模光纤与直观导光分析 | 单模光纤、少模光纤与模式分析 |
        | 可观察现象 | 入射角过大导致不能有效耦合 | V 参数变化导致模式数和截止状态变化 |
        | 教学作用 | 帮助学生理解“光为什么被约束在纤芯中” | 帮助学生理解“光纤为什么有模式和截止波长” |
        | 局限性 | 无法解释真实电磁场模式分布 | 数学模型较复杂，教学中宜采用可视化近似 |
        """)

        st.markdown("#### 自动生成实验结论")
        conclusion = """# 光纤传输原理虚拟仿真实验报告

## 一、实验目的
理解阶跃型光纤的导光条件，掌握全反射、数值孔径、最大接收角、V 参数、单模条件和截止波长之间的关系。

## 二、实验原理
几何光学方法把光看作射线，重点分析纤芯/包层界面处的全反射条件；波动理论把光看作电磁波，重点分析归一化频率 V 与 LP 模式分布。

## 三、实验任务
1. 调节纤芯和包层折射率，观察 NA 与最大接收角的变化。
2. 调节入射角，判断是否满足光纤接收条件。
3. 调节波长和纤芯半径，观察 V 参数与单模/多模状态变化。
4. 对比 LP01、LP11、LP21 等模式场强分布。

## 四、实验结论
几何光学方法适合解释光纤导光的直观条件，波动理论适合解释模式、截止波长和单模传输条件。二者结合可以形成从基础物理原理到光纤通信系统仿真的完整教学链条。
"""
        st.download_button("📥 下载本实验报告模板", conclusion, "Fiber_Transmission_Principle_Report.md", mime="text/markdown")
        st.info("建议教学安排：该实验可放在光纤损耗、带宽、截止波长和色散测量实验之前，作为理论前置实验。")

def render_tab_edfa():
    st.header("🔋 EDFA 掺铒光纤放大器工程仿真")
    st.markdown("该模块由原经验增益曲线升级为小信号增益、增益饱和、噪声指数和 ASE 噪声共同作用的工程近似模型。")
    c1, c2 = st.columns([1, 2])
    with c1:
        signal_in = st.number_input("输入信号功率 (dBm)", value=-20.0, step=1.0)
        pump_power = st.slider("泵浦功率 (mW) @ 980nm", 10.0, 500.0, 100.0, 10.0)
        edf_length = st.slider("掺铒光纤长度 (m)", 1.0, 30.0, 10.0, 1.0)
        nf_db = st.slider("噪声指数 NF (dB)", 3.5, 8.0, 5.5, 0.1)
        psat_out_dbm = st.slider("饱和输出功率 Psat,out (dBm)", 5.0, 25.0, 17.0, 0.5)
        optical_bw_ghz = st.slider("等效光滤波带宽 (GHz)", 12.5, 100.0, 25.0, 12.5)

    with c2:
        h = 6.62607015e-34
        c = 2.99792458e8
        nu = c / 1550e-9
        p_in_w = 10 ** ((signal_in - 30) / 10.0)
        psat_w = 10 ** ((psat_out_dbm - 30) / 10.0)

        # 泵浦和光纤长度共同决定小信号增益；长度过短吸收不充分，过长反吸收和 ASE 增强。
        g0_db = 34.0 * (1 - np.exp(-pump_power / 120.0)) * (1 - np.exp(-edf_length / 5.5)) * np.exp(-edf_length / 42.0)
        g0_lin = 10 ** (g0_db / 10.0)
        gain_lin = g0_lin / (1.0 + (p_in_w * g0_lin) / max(psat_w, 1e-15))
        gain_db = 10 * np.log10(max(gain_lin, 1e-12))
        out_signal_dbm = signal_in + gain_db

        nf_lin = 10 ** (nf_db / 10.0)
        n_sp = nf_lin / 2.0
        optical_bw = optical_bw_ghz * 1e9
        p_ase_w = 2.0 * n_sp * h * nu * max(gain_lin - 1.0, 0.0) * optical_bw
        p_ase_dbm = 10 * np.log10(max(p_ase_w, 1e-24) * 1e3)
        p_out_w = 10 ** ((out_signal_dbm - 30) / 10.0)
        osnr_out_db = 10 * np.log10(max(p_out_w / max(p_ase_w, 1e-24), 1e-12))

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("小信号增益 G0", f"{g0_db:.2f} dB")
        col_m2.metric("饱和后实际增益", f"{gain_db:.2f} dB")
        col_m3.metric("输出信号功率", f"{out_signal_dbm:.2f} dBm")
        col_m4, col_m5, col_m6 = st.columns(3)
        col_m4.metric("ASE 噪声功率", f"{p_ase_dbm:.2f} dBm")
        col_m5.metric("输出 OSNR", f"{osnr_out_db:.2f} dB")
        col_m6.metric("增益压缩", f"{max(0.0, g0_db - gain_db):.2f} dB")

        pin_sweep_dbm = np.linspace(-40, 10, 300)
        pin_sweep_w = 10 ** ((pin_sweep_dbm - 30) / 10.0)
        gain_sweep_lin = g0_lin / (1.0 + (pin_sweep_w * g0_lin) / max(psat_w, 1e-15))
        gain_sweep_db = 10 * np.log10(np.maximum(gain_sweep_lin, 1e-12))
        pout_sweep_dbm = pin_sweep_dbm + gain_sweep_db

        fig_edfa, ax_g = plt.subplots(figsize=(8, 4))
        ax_g.plot(pin_sweep_dbm, gain_sweep_db, linewidth=2, label='实际增益')
        ax_g.axhline(g0_db, linestyle='--', alpha=0.7, label='小信号增益')
        ax_g.axvline(signal_in, linestyle=':', alpha=0.7, label='当前输入')
        ax_g.set_xlabel("输入信号功率 (dBm)")
        ax_g.set_ylabel("增益 (dB)")
        ax_g.set_title("EDFA 增益饱和曲线")
        ax_g.legend(loc='upper right')
        st.pyplot(fig_edfa)

        fig_pout, ax_pout = plt.subplots(figsize=(8, 3.5))
        ax_pout.plot(pin_sweep_dbm, pout_sweep_dbm, linewidth=2, label='输出功率')
        ax_pout.axhline(psat_out_dbm, linestyle='--', alpha=0.7, label='Psat,out')
        ax_pout.scatter([signal_in], [out_signal_dbm], s=60, zorder=5, label='当前工作点')
        ax_pout.set_xlabel("输入信号功率 (dBm)")
        ax_pout.set_ylabel("输出信号功率 (dBm)")
        ax_pout.set_title("EDFA 输出功率与饱和限制")
        ax_pout.legend(loc='lower right')
        st.pyplot(fig_pout)

        if g0_db - gain_db > 3:
            st.warning("⚠️ 放大器已出现明显增益压缩，真实实验中应降低输入功率、提高饱和输出功率或调整泵浦/EDF长度。")
        if osnr_out_db < 20:
            st.warning("⚠️ 输出 OSNR 偏低，ASE 噪声会明显影响后级接收性能。")
        st.info("模型说明：该 EDFA 模型用于工程趋势分析，未求解完整铒离子速率方程；论文中可称为‘工程近似模型’。")

@st.fragment
def render_tab_otdr(client, active_model_id):
    st.header("📉 OTDR 光缆故障定位与 AI 分析实验")
    st.markdown("该模块加入脉冲宽度、动态范围、事件死区、平均次数和后向散射噪声，更接近真实 OTDR 台架实验。")
    col_o1, col_o2 = st.columns([1, 3])
    with col_o1:
        st.subheader("1. 实验设置")
        fiber_len = st.slider("光纤总长度 (km)", 10, 100, 50)
        otdr_alpha = st.slider("光纤衰减系数 (dB/km)", 0.15, 0.40, 0.20, 0.01)
        pulse_width_ns = st.select_slider("OTDR 脉冲宽度 (ns)", options=[5, 10, 30, 50, 100, 300, 1000], value=50)
        dynamic_range_db = st.slider("动态范围 (dB)", 25.0, 50.0, 38.0, 1.0)
        avg_count = st.select_slider("平均次数", options=[1, 4, 16, 64, 256], value=16)
        mode = st.radio("实验模式", ["手动设置故障", "🎲 盲测模式 (随机故障)"])

        events_list = []
        if mode == "手动设置故障":
            if st.checkbox("在 15km 处添加活动连接器"): events_list.append({'pos': 15, 'type': 'connector'})
            if st.checkbox("在 25km 处添加熔接点"): events_list.append({'pos': 25, 'type': 'fusion'})
            if st.checkbox("在 30km 处添加宏弯损耗"): events_list.append({'pos': 30, 'type': 'bend'})
            if st.checkbox("在 45km 处光纤断裂"): events_list.append({'pos': 45, 'type': 'break'})
        else:
            if st.button("🎲 生成随机链路"):
                num_faults = random.randint(1, 3)
                possible_types = ['connector', 'fusion', 'bend', 'break']
                st.session_state.random_events = []
                positions = sorted(random.sample(range(5, max(6, fiber_len-5)), num_faults))
                for p in positions:
                    st.session_state.random_events.append({'pos': p, 'type': random.choice(possible_types)})
            if 'random_events' in st.session_state:
                events_list = st.session_state.random_events

    with col_o2:
        x_data, y_data, true_labels = simulate_otdr_trace(
            fiber_len, events_list,
            attenuation_coeff=otdr_alpha,
            pulse_width_ns=float(pulse_width_ns),
            dynamic_range_db=dynamic_range_db,
            avg_count=avg_count
        )
        resolution_m = 3e8 * float(pulse_width_ns) * 1e-9 / (2 * 1.468)
        st.caption(f"空间分辨率约 {resolution_m:.1f} m；脉冲越宽，动态范围越高但事件死区越大。")

        fig_otdr, ax_otdr = plt.subplots(figsize=(8, 4))
        ax_otdr.plot(x_data, y_data, linewidth=1.5, label='OTDR 后向散射迹线')
        ax_otdr.axhline(-dynamic_range_db, linestyle=':', alpha=0.7, label='噪声底/动态范围')
        ax_otdr.set_title("OTDR 后向散射曲线与事件响应")
        ax_otdr.set_xlabel("距离 (km)")
        ax_otdr.set_ylabel("相对后向散射电平 (dB)")
        ax_otdr.set_ylim(min(y_data)-3, max(y_data)+4)
        ax_otdr.legend(loc='upper right')
        st.pyplot(fig_otdr)

        st.subheader("🧠 AI 智能诊断助理")
        col_ai1, col_ai2 = st.columns(2)
        with col_ai1:
            if st.button("🔍 启动 AI 自动分析"):
                with st.spinner("AI 正在扫描波形特征..."):
                    time.sleep(0.5)
                    df_events = ai_analyze_otdr(y_data, x_data, pulse_width_ns=float(pulse_width_ns), dynamic_range_db=dynamic_range_db)
                    if not df_events.empty:
                        st.dataframe(df_events, hide_index=True)
                        prompt = f"AI 算法检测到的 OTDR 事件表如下：\n{df_events.to_string()}\n请根据这些数据，简要生成一份'故障诊断报告'，并说明脉冲宽度、事件死区对定位精度的影响。"
                        if client is None:
                            show_ai_unavailable("OTDR AI 诊断")
                        else:
                            try:
                                completion = client.chat.completions.create(
                                    model=active_model_id,
                                    messages=[{"role": "system", "content": "你是一名专业的光通信工程师。"}, {"role": "user", "content": prompt}],
                                )
                                st.success("✅ AI 诊断报告生成成功")
                                st.markdown(completion.choices[0].message.content)
                            except Exception:
                                st.error("无法连接 AI 导师，仅显示检测数据。")
                    else:
                        st.success("✅ 链路检测正常，未发现明显故障点。")
        with col_ai2:
            st.markdown("#### 实验参数解释")
            st.info("脉冲宽度决定空间分辨率与死区；动态范围决定最长可测距离；平均次数越高噪声越低但测试时间越长。")
            if mode == "🎲 盲测模式 (随机故障)":
                with st.expander("🔐 教师/答案模式 (点击查看真值)"):
                    for label in true_labels:
                        st.code(label)

@st.fragment
def render_tab2_ai_diag(params, metrics, client, active_model_id):
    st.header("AI 智能故障诊断系统")
    col_t1, col_t2 = st.columns([1, 2])
    telemetry = f"- 波长: {params['wavelength']}\n- 调制: {params['modulation']}\n- 距离: {params['distance']} km\n- Rx功率: {metrics['rx_power']:.2f} dBm\n- OSNR: {metrics['osnr']:.2f} dB\n- Q因子: {metrics['q_factor']:.2f}\n- BER: {metrics['ber']:.2e}"
    with col_t1: st.markdown("#### 📡 实时遥测数据"); st.code(telemetry)
    with col_t2:
        st.markdown("#### 💬 AI 专家诊断意见")
        if st.button("🚀 运行 AI 诊断"):
            if client is None: show_ai_unavailable("AI 故障诊断")
            else:
                with st.spinner("AI 正在思考中..."):
                    try:
                        prompt = f"当前仿真使用 {params['wavelength']} 和 {params['modulation']}。\n遥测数据:\n{telemetry}\n请分析。"
                        response = client.chat.completions.create(
                            model=active_model_id,
                            messages=[{"role": "system", "content": "你是一名严谨的上海交通大学光纤通信实验室助教。"}, {"role": "user", "content": prompt}],
                        )
                        st.markdown(response.choices[0].message.content)
                    except Exception as e: st.error(f"连接本地模型失败: {e}")

@st.fragment
def render_tab4_coding(metrics):
    st.header("🔣 线路编码实验")

    st.markdown("### ⚙️ 编码实验参数")
    col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 2])
    with col_cfg1:
        code_type = st.radio("编码类型", ["CMI", "5B6B"], key="tab4_code_type")
    with col_cfg2:
        input_method = st.radio("数据来源", ["随机生成", "手动输入"], key="tab4_input_method")
    with col_cfg3:
        if input_method == "随机生成":
            data_length = st.slider("数据位长度", 10, 50, 20, 5, key="tab4_data_length")
            if st.button("🔄 刷新随机比特流", key="tab4_refresh_bits"):
                st.session_state.random_bits = [random.randint(0, 1) for _ in range(data_length)]
            raw_bits = st.session_state.random_bits[:data_length]
        else:
            user_input = st.text_input("输入二进制串 (0/1)", "1011001010", key="tab4_user_bits")
            clean_input = [int(c) for c in user_input if c in ['0', '1']]
            raw_bits = clean_input if clean_input else [1, 0, 1, 0]

    if code_type == "5B6B":
        st.info("5B6B 当前采用运行数字受控的工程化教学编码：按 5 bit 分组映射为 6 bit，优先选择直流平衡、短连码码字，可展示真实 5B6B 实验中常见的码率开销、运行数字控制、长连码抑制与接收波形退化趋势。")

    use_manual = st.checkbox("启用手动调整模式 (覆盖 Tab 1 物理计算结果)", value=False, key="tab4_manual_mode")
    
    if use_manual:
        col_m1, col_m2 = st.columns(2)
        with col_m1: manual_osnr = st.slider("📉 信噪比 (OSNR, dB)", 0.0, 35.0, 15.0, 0.5, key="tab4_manual_osnr")
        with col_m2: manual_bw = st.slider("📶 带宽限制因子", 0.05, 1.0, 0.2, 0.05, key="tab4_manual_bw")
        current_osnr, current_bw = manual_osnr, manual_bw
        st.info(f"🔧 手动模式生效: OSNR={current_osnr}dB, Bandwidth Factor={current_bw}")
    else:
        current_osnr, current_bw = metrics['osnr'], 0.2 
        st.caption(f"🤖 自动模式: OSNR={current_osnr:.2f}dB，接收波形由链路仿真结果驱动")

    st.markdown("##### 🔢 当前输入比特流:")
    st.code("".join([str(b) for b in raw_bits]), language="text")
    
    encoded_bits = encode_cmi(raw_bits) if code_type == "CMI" else encode_5b6b(raw_bits)
    t, rx_signal, ideal_signal = apply_channel_effects(encoded_bits, current_osnr, bandwidth_factor=current_bw)

    def _max_run(seq):
        if not seq: return 0
        best = cur = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                cur += 1; best = max(best, cur)
            else:
                cur = 1
        return best

    def _transition_density(seq):
        if len(seq) < 2: return 0.0
        return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1]) / (len(seq) - 1)

    ones = sum(encoded_bits)
    zeros = len(encoded_bits) - ones
    disparity = ones - zeros
    code_rate = len(raw_bits) / max(len(encoded_bits), 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("编码后码元数", f"{len(encoded_bits)}")
    c2.metric("有效码率", f"{code_rate:.3f}")
    c3.metric("运行数字差", f"{disparity:+d}")
    c4.metric("最长连码", f"{_max_run(encoded_bits)} bit")

    if code_type == "5B6B":
        st.caption("5B6B 理论有效码率约为 5/6≈0.833；运行数字差越接近 0、最长连码越短，越接近实际线路编码对直流平衡和时钟恢复的要求。")
    
    fig3, (ax_orig, ax_code, ax_rx) = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    plt.subplots_adjust(hspace=0.5)
    ax_orig.step(range(len(raw_bits)), raw_bits, where='post', color='#00FF00', linewidth=2)
    ax_orig.set_title("1. 原始二进制数据 (NRZ)", color='white')
    ax_orig.set_ylim(-0.2, 1.2)
    x_code = np.linspace(0, len(raw_bits), len(encoded_bits))
    ax_code.step(x_code, encoded_bits, where='post', color='#00FFFF', linewidth=2)
    ax_code.set_title(f"2. {code_type} 编码后信号", color='white')
    ax_code.set_ylim(-0.2, 1.2)
    x_rx = np.linspace(0, len(raw_bits), len(rx_signal))
    ax_rx.plot(x_rx, rx_signal, color='#FF4B4B', linewidth=1.5)
    mode_str = "手动调试" if use_manual else "物理仿真"
    ax_rx.set_title(f"3. 接收波形 ({mode_str}: OSNR={current_osnr:.1f}dB, BW={current_bw})", color='white')
    st.pyplot(fig3)

    threshold = 0.5
    sample_offset = 10
    samples_per_bit = max(1, int(len(rx_signal) / max(len(encoded_bits), 1)))
    sample_indices = np.clip(np.arange(len(encoded_bits)) * samples_per_bit + sample_offset, 0, len(rx_signal) - 1).astype(int)
    decided_bits = (rx_signal[sample_indices] >= threshold).astype(int).tolist()
    line_errors = sum(int(a != b) for a, b in zip(decided_bits, encoded_bits))
    measured_ber = line_errors / max(len(encoded_bits), 1)
    eye_q = (np.mean(rx_signal[sample_indices][np.array(encoded_bits) == 1]) - np.mean(rx_signal[sample_indices][np.array(encoded_bits) == 0])) / (np.std(rx_signal[sample_indices]) + 1e-12) if len(set(encoded_bits)) > 1 else 0.0
    e1, e2, e3 = st.columns(3)
    e1.metric("线路判决误码率", f"{measured_ber:.2e}")
    e2.metric("跳变密度", f"{_transition_density(encoded_bits):.2f}")
    e3.metric("眼图裕量估计", f"{eye_q:.2f}")

    st.markdown("---"); st.markdown("### 🆚 深度对比: 1B2B (CMI) vs 5B6B")
    bits_cmi, bits_5b6b = encode_cmi(raw_bits), encode_5b6b(raw_bits)
    fig_cmp, (ax_cmi, ax_5b6b) = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(hspace=0.4)
    t_cmi = np.linspace(0, len(raw_bits), len(bits_cmi) + 1)
    ax_cmi.step(t_cmi[:-1], bits_cmi, where='post', color='#00FFFF', linewidth=1.5)
    ax_cmi.set_title(f"1B2B (CMI) 编码 | 码元数: {len(bits_cmi)} | 码率={len(raw_bits)/max(len(bits_cmi),1):.3f}", color='white')
    ax_cmi.set_ylim(-0.2, 1.2)
    t_5b6b = np.linspace(0, len(raw_bits), len(bits_5b6b) + 1)
    ax_5b6b.step(t_5b6b[:-1], bits_5b6b, where='post', color='#FF00FF', linewidth=1.5)
    ax_5b6b.set_title(f"5B6B 编码 | 码元数: {len(bits_5b6b)} | 码率={len(raw_bits)/max(len(bits_5b6b),1):.3f}", color='white')
    ax_5b6b.set_ylim(-0.2, 1.2); ax_5b6b.set_xlabel("Normalized Time (Bit Periods)")
    st.pyplot(fig_cmp)

@st.fragment
def render_tab_phone(metrics, params):
    st.header("📞 电话语音光纤传输仿真 (PCM + 光链路)")
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        audio_source = st.radio("语音数据源", ["合成测试音 (双频正弦波)", "单频基准音 (1kHz)"])
        duration = st.slider("信号时长 (秒)", 0.1, 2.0, 0.5, 0.1)
        fs, bits_per_sample, levels = 8000, 8, 256
        current_ber = metrics['ber']
        st.info(f"🔗 当前光链路物理状态：\n- 距离: {params['distance']} km\n- BER: {current_ber:.2e}")
        
    with col_p2:
        t_audio = np.linspace(0, duration, int(fs * duration), endpoint=False)
        if audio_source == "单频基准音 (1kHz)": analog_sig = np.sin(2 * np.pi * 1000 * t_audio)
        else: analog_sig = 0.6 * np.sin(2 * np.pi * 1000 * t_audio) + 0.4 * np.sin(2 * np.pi * 2500 * t_audio)
            
        quantized_sig = np.clip(np.round((analog_sig + 1.0) / 2.0 * (levels - 1)), 0, levels - 1).astype(int)
        tx_bits = np.unpackbits(quantized_sig.astype(np.uint8))
        error_mask = np.random.rand(len(tx_bits)) < current_ber
        rx_bits = tx_bits ^ error_mask.astype(np.uint8)
        rx_quantized = np.packbits(rx_bits)
        recovered_sig = (rx_quantized / (levels - 1)) * 2.0 - 1.0
        
        fig_phone, ax_phone = plt.subplots(figsize=(8, 3))
        show_len = min(100, len(t_audio))
        ax_phone.plot(t_audio[:show_len]*1000, analog_sig[:show_len], color='#00FF00', label='Tx Original (Analog)', alpha=0.8)
        ax_phone.step(t_audio[:show_len]*1000, recovered_sig[:show_len], color='#FF4B4B', label='Rx Recovered (DAC)', where='mid', alpha=0.8)
        ax_phone.set_xlabel("Time (ms)"); ax_phone.set_ylabel("Amplitude")
        ax_phone.set_title("PCM Encoding & Optical Transmission Result")
        ax_phone.legend(); ax_phone.grid(True, alpha=0.3)
        st.pyplot(fig_phone)

    st.markdown("---"); st.subheader("🎧 3. 听觉评估 (Subjective Verification)")
    col_audio1, col_audio2 = st.columns(2)
    with col_audio1:
        st.markdown("**发送端 (原始纯净语音)**")
        st.audio(np.int16(analog_sig * 32767), sample_rate=fs)
    with col_audio2:
        st.markdown(f"**接收端 (经过 {params['distance']}km 光纤传输)**")
        st.audio(np.int16(recovered_sig * 32767), sample_rate=fs)

@st.fragment
def render_tab_design(client, active_model_id):
    st.header("🏆 前沿光通信系统综合设计挑战")
    scenario = st.selectbox("📍 选择业务场景 (Industry Scenarios)", ["场景 A：【东数西算】1000km 超长距骨干网设计", "场景 B：【智算中心】400G DCI 数据中心互联", "场景 C：【5G 前传】高分光比 WDM-PON 接入网"])
    st.markdown("---")
    with st.form("design_form_pro"):
        if "东数西算" in scenario:
            c1, c2 = st.columns(2)
            with c1: d_total, span_len, tx_p_ch = st.number_input("总传输距离 (km)", value=1000, disabled=True), st.slider("单跨段长度 (km)", 50, 120, 80, 10), st.slider("单波入纤功率 (dBm)", -10.0, 10.0, 0.0, 0.5)
            with c2: amp_type, use_dcf, fec_type = st.radio("中继放大器", ["纯 EDFA", "混合放大"]), st.checkbox("部署 DCF", value=True), st.selectbox("FEC", ["标准 RS-FEC", "软判决 SD-FEC"])
        elif "智算中心" in scenario:
            c1, c2 = st.columns(2)
            with c1: d_dci, mod_dci, tx_p_dci = st.slider("物理间距 (km)", 10, 120, 80, 5), st.selectbox("调制格式", ["QPSK", "16-QAM", "64-QAM"]), st.slider("模块输出功率 (dBm)", -15.0, 5.0, -5.0, 0.5)
            with c2: fiber_att, conn_loss, rx_sens = st.slider("光纤衰减 (dB/km)", 0.18, 0.25, 0.20, 0.01), st.number_input("接头损耗 (dB)", value=3.5), st.number_input("接收机灵敏度 (dBm)", value=-20.0)
        else:
            c1, c2 = st.columns(2)
            with c1: d_pon, split_ratio, awg_loss = st.slider("ODN半径 (km)", 5, 40, 20, 1), st.selectbox("分光比", ["1:16", "1:32", "1:64", "1:128"]), st.slider("AWG插损 (dB)", 3.0, 8.0, 5.0)
            with c2: olt_tx, onu_rx_req = st.slider("OLT发射功率 (dBm)", 0.0, 15.0, 5.0, 0.5), st.number_input("ONU灵敏度 (dBm)", value=-25.0)
        submitted_pro = st.form_submit_button("🚀 提交系统级架构设计至 AI 导师评估")

    if submitted_pro:
        with st.spinner("AI 导师正在综合评估复杂的工业级设计方案..."):
            if "东数西算" in scenario: prompt = f"东数西算设计: {span_len}km/跨, {tx_p_ch}dBm, {amp_type}, {fec_type}。请评估。"
            elif "智算中心" in scenario: prompt = f"DCI互联: {d_dci}km, {mod_dci}, {tx_p_dci}dBm。请评估。"
            else: prompt = f"PON设计: {d_pon}km, 分光比{split_ratio}, OLT {olt_tx}dBm, ONU要求 {onu_rx_req}dBm。请评估。"
            if client is None: show_ai_unavailable("工业场景架构评估")
            else:
                try:
                    response = client.chat.completions.create(model=active_model_id, messages=[{"role": "system", "content": "你是光通信行业的顶级技术评审专家。"}, {"role": "user", "content": prompt}])
                    st.success("✅ 架构评估完成！"); st.write(response.choices[0].message.content)
                except Exception: st.error("AI 评估系统未连接。")
    
    with st.form("design_form"):
        d_dist, d_tx, d_mod = st.slider("总传输距离 (km)", 50, 300, 100, 10), st.slider("发射机功率 (dBm)", -10.0, 20.0, 0.0), st.selectbox("调制格式2", ["OOK", "QPSK", "16-QAM"])
        use_edfa, use_dcf = st.checkbox("部署 EDFA"), st.checkbox("部署 DCF色散补偿")
        if st.form_submit_button("🚀 提交设计方案至 AI 导师审核"):
            rx_p = d_tx - d_dist * 0.2 + (20 if use_edfa else 0)
            if client is None: show_ai_unavailable("设计方案评审")
            else:
                with st.spinner("正在评估..."):
                    try:
                        response = client.chat.completions.create(model=active_model_id, messages=[{"role": "user", "content": f"系统设计: 距离{d_dist}km, 功率{d_tx}dBm, 调制{d_mod}。请点评。"}])
                        st.success("✅ 评估完成！"); st.write(response.choices[0].message.content)
                    except Exception: st.error("AI未连接。")

@st.fragment
def render_tab3_chat(client, active_model_id, params):
    st.header("👩‍🏫 AI 实验导师")
    if client is None: st.warning("当前为规则库/离线模式，聊天导师暂不启用；其他仿真实验和规则库报告可正常使用。")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("关于光通信有什么问题？", disabled=client is None):
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            if client is None: show_ai_unavailable("AI 实验导师")
            else:
                try:
                    stream = client.chat.completions.create(
                        model=active_model_id,
                        messages=[{"role": "system", "content": f"你是一位教授。当前参数: 波长={params['wavelength']}, 调制={params['modulation']}。"}, *st.session_state.chat_history],
                        stream=True,
                    )
                    response = st.write_stream(stream)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e: st.error(f"调用失败: {e}")

# ==========================================
# 6. 渲染界面调用
# ==========================================
with tab1: render_tab1_link_sim(params, metrics, client, active_model_id)
with tab5: render_tab5_device_test(params)
with tab_ld: render_tab_ld()
with tab_fiber_param: render_tab_fiber_param()
with tab_fiber_principle: render_tab_fiber_principle()
with tab_edfa: render_tab_edfa()
with tab_otdr: render_tab_otdr(client, active_model_id)
with tab2: render_tab2_ai_diag(params, metrics, client, active_model_id)
with tab4: render_tab4_coding(metrics)
with tab_phone: render_tab_phone(metrics, params)
with tab_design: render_tab_design(client, active_model_id)
with tab3: render_tab3_chat(client, active_model_id, params)