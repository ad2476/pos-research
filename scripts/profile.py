import pstats

p = pstats.Stats('../debug/stats')
p.strip_dirs().sort_stats('tottime').print_stats()

