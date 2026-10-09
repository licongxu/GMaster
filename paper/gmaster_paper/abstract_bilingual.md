# Bilingual abstract (sidecar; MNRAS body is English only)

## English

Pseudo-$C_\ell$ estimation on HEALPix maps is dominated by the latitudinal spherical-harmonic transform and the MASTER mode-coupling matrix. GMaster implements three exact reductions of those operators, matching the NaMaster public API on GPU. Hermitian half-spectrum chirp-$Z$ synthesis halves the polar-cap convolution. The equatorial belt (two thirds of a RING map) is an ordinary FFT of length $4N_{\mathrm{side}}$. Offset-blocked scalar coupling visits $\sim n^3/3$ cells. With float32 storage of float64-generated Legendre tables, scalar $N_{\mathrm{side}}=1024$ runs in $677\,\mathrm{ms}$ against NaMaster at $1709\,\mathrm{ms}$ ($2.53\times$), and isolated scalar transforms beat DUCC in all six spin-$0$ cells tested. Float32 table storage costs $1.1\times10^{-7}$ relative rms. Polarised $N_{\mathrm{side}}=1024$ does not fit in a $71\,\mathrm{GiB}$ pool.

Keywords: spherical harmonic transform; HEALPix; chirp-$Z$; MASTER; mode-coupling matrix; GPU; associated Legendre functions

## 繁體中文

HEALPix 地圖上的偽 $C_\ell$ 估計由緯向球諧變換與 MASTER 模耦合矩陣主導。GMaster 在 GPU 上實作這兩類算子的三項精確約化，並對齊 NaMaster 公開 API。厄米半譜 chirp-$Z$ 綜合將極蓋捲積長度減半；赤道帶（RING 地圖的三分之二）是長度 $4N_{\mathrm{side}}$ 的普通 FFT；位移分塊的純量耦合只造訪約 $n^3/3$ 個單元。以 float64 生成、float32 儲存的 Legendre 表，純量 $N_{\mathrm{side}}=1024$ 端到端 $677\,\mathrm{ms}$，對 NaMaster $1709\,\mathrm{ms}$（$2.53\times$）；獨立純量變換在全部六個 spin-$0$ 格子上快於 DUCC。float32 表儲存的相對 rms 為 $1.1\times10^{-7}$。偏振 $N_{\mathrm{side}}=1024$ 的 Wigner-$d$ 佈局無法裝進 $71\,\mathrm{GiB}$ 池。

關鍵詞：球諧變換；HEALPix；chirp-$Z$；MASTER；模耦合矩陣；GPU；關聯勒讓德函數
