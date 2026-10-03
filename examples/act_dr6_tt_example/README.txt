MAP=/rds/datasets/act/act_dr6.02_maps_standard/act_dr6.02_std_AA_night_pa4_f150_4way_coadd_map_srcfree_healpix.fits
CACHE=.qwen/tmp/act/act_{T,w}_nside4096.npy (examples/act_dr6_prepare.py)
MASK=finite & nonzero pixels of the map (footprint), same array for both estimators, f_sky=0.4807
nside=4096 lmax=12287 nlb=50 n_iter=3 spin=0
t_gmaster=7.628s (runs: 213.249, 7.628; first includes JIT compilation)
t_namaster=70.433s (cores=192)
ratio GM/NM-1: rms=1.414e-05 max|.|=4.916e-05
timed: field + coupling matrix + coupled cell + decouple; I/O excluded
