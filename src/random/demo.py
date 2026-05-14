import numpy as np
import matplotlib.pyplot as plt
from ccd_simulator import SimulatorConfig, simulate_ccd
from ccd_decoder import DecoderConfig, decode_ccd

sim_cfg = SimulatorConfig(n_bits=24, bit_width_px=7.5, sigma_blur_px=1.5, oversample=32, n_pixels=256, adc_bits=12, noise_sigma_adu=60.0, vignette_strength=0.25, seed=7)
dec_cfg = DecoderConfig(bit_width_hint_px=sim_cfg.bit_width_px, fir_order=8, edge_threshold_rel=0.25, adaptive_threshold=True, roi_threshold_rel=0.22, roi_margin_bits=1.5)

sim = simulate_ccd(sim_cfg)
dec = decode_ccd(sim.adc_signal, dec_cfg, sim.true_edges, sim.bits, sim_cfg.bit_width_px)

print('=' * 58)
print('  Параметры генерации')
print(f'  N_BITS={sim_cfg.n_bits}, BIT_WIDTH={sim_cfg.bit_width_px} px, σ={sim_cfg.sigma_blur_px} px')
print(f'  noise={sim_cfg.noise_sigma_adu} adu, vignette={sim_cfg.vignette_strength}')
print('=' * 58)
print(f'  ROI:              [{dec.roi_bounds_px[0]:.2f}, {dec.roi_bounds_px[1]:.2f}] px')
print(f'  Истинные биты:    {"".join(map(str, sim.bits))}')
print(f'  Декодированные:   {"".join(map(str, dec.decoded_bits))}')
print(f'  Точность:         {dec.accuracy*100:.1f}%' if dec.accuracy is not None else '  Точность: N/A')
print(f'  T оценённый:      {dec.clock_period_px:.4f} пикс.  (истинный: {sim_cfg.bit_width_px})')
print(f'  Ошибка T:         {dec.clock_period_error_px:+.4f} пикс.')
print(f'  Найдено фронтов:  {dec.n_edges_found} / {dec.n_edges_true} в ROI')
print(f'  RMSE позиций:     {dec.edge_rmse_px:.4f} пикс.' if dec.edge_rmse_px is not None else '  RMSE: N/A')
print(f'  Bias позиций:     {dec.edge_bias_px:+.4f} пикс.' if dec.edge_bias_px is not None else '  Bias: N/A')
print('=' * 58)

n_px = sim_cfg.n_pixels
px = np.arange(n_px)
adc_max = (1 << sim_cfg.adc_bits) - 1
fig = plt.figure(figsize=(14, 15))
gs = fig.add_gridspec(5, 1, hspace=0.48)
axes = [fig.add_subplot(gs[i]) for i in range(5)]

ax = axes[0]
ax.plot(sim.x_continuous, sim.profile_continuous, color='steelblue', lw=0.8, label='Профиль (после Гаусса)')
for te in sim.true_edges:
    ax.axvline(te, color='green', lw=0.6, alpha=0.45)
ax.axvspan(dec.roi_bounds_px[0], dec.roi_bounds_px[1], color='gold', alpha=0.12, label='ROI кода')
ax.set_xlim(0, n_px); ax.legend(fontsize=8)
ax.set_title('Непрерывный профиль')

ax = axes[1]
ax.imshow(sim.adc_signal[np.newaxis, :], cmap='gray', aspect='auto', vmin=0, vmax=adc_max, extent=[0, n_px, 0, 1])
ax.axvspan(dec.roi_bounds_px[0], dec.roi_bounds_px[1], color='gold', alpha=0.18)
ax.set_yticks([])
ax.set_title('Изображение ПЗС-линейки')

ax = axes[2]
ax.plot(px, dec.signal_normalized, 'k', lw=1.0, label='Сигнал (норм.)')
ax.axhline(dec.threshold, ls='--', color='gray', lw=0.8, label=f'Порог = {dec.threshold:.3f}')
ax.axvspan(dec.roi_bounds_px[0], dec.roi_bounds_px[1], color='gold', alpha=0.12, label='ROI')
for te in sim.true_edges[1:-1]:
    ax.axvline(te, color='green', lw=0.7, alpha=0.25)
for ep in dec.edge_positions_px:
    ax.axvline(ep, color='red', lw=0.8, alpha=0.55)
for bc, db in zip(dec.bit_centers_px, dec.decoded_bits):
    col = 'royalblue' if db else 'darkorange'
    ax.plot(bc, np.interp(bc, px, dec.signal_normalized), 'o', color=col, ms=4.5, alpha=0.85)
ax.set_xlim(0, n_px); ax.legend(fontsize=7, ncol=2)
ax.set_title('ROI, фронты, центры бит')

ax = axes[3]
ax.plot(px, dec.derivative, color='purple', lw=0.9, label='Производная (КИХ)')
ax.plot(px, np.abs(dec.derivative), color='orange', lw=0.7, alpha=0.7, label='|Производная|')
ax.axvspan(dec.roi_bounds_px[0], dec.roi_bounds_px[1], color='gold', alpha=0.12)
for ep in dec.edge_positions_px:
    ax.axvline(ep, color='red', lw=0.8, alpha=0.55)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlim(0, n_px); ax.legend(fontsize=7)
ax.set_title('Фронты по нулевым переходам производной')

ax = axes[4]
inner_true = sim.true_edges[1:-1]
inner_true = inner_true[(inner_true >= dec.roi_bounds_px[0]) & (inner_true <= dec.roi_bounds_px[1])]
ax.vlines(inner_true, 0, 1, colors='green', lw=2, alpha=0.55, label='Истинные фронты')
ax.vlines(dec.edge_positions_px, 1.25, 2.25, colors='red', lw=2, alpha=0.55, label='Найденные фронты')
for bc, db in zip(dec.bit_centers_px, dec.decoded_bits):
    col = 'royalblue' if db else 'darkorange'
    ax.text(bc, 2.65, str(db), ha='center', va='center', fontsize=7, color='white', fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc=col, ec='none'))
acc_str = f'{dec.accuracy*100:.1f}%' if dec.accuracy is not None else 'N/A'
rms_str = f'{dec.edge_rmse_px:.4f} пикс.' if dec.edge_rmse_px is not None else 'N/A'
ax.set_ylim(-0.3, 3.3)
ax.set_yticks([0.5, 1.75, 2.65], ['Истинные', 'Найденные', 'Биты'])
ax.set_xlim(dec.roi_bounds_px[0]-5, dec.roi_bounds_px[1]+5)
ax.legend(fontsize=8)
ax.set_title(f'Позиции фронтов в ROI | RMSE={rms_str} | Точность: {acc_str}')

fig.suptitle('ПЗС-симулятор + декодирование по ROI и нулевым переходам производной', fontsize=11, fontweight='bold')
plt.savefig('~/output/ccd_demo_result.png'.replace('~','/root'), dpi=150, bbox_inches='tight')
print('saved')
