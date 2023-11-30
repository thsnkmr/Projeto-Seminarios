def fazerAva(ObjSeg, GS):

	import numpy as np
    
	ObjSeg = ObjSeg > 0.5
	GS = GS > 0.5

	(M, N) = np.shape(ObjSeg)

	AreaImagem = M*N
	AreaInter = np.sum((ObjSeg*GS))
	AreaSeg = np.sum(ObjSeg)
	AreaGS = np.sum(GS)

	VP = 100*(AreaInter/AreaGS)
	FP = 100*(AreaSeg - AreaInter)/(AreaImagem - AreaGS)
	FN = 100*(AreaGS - AreaInter)/(AreaGS)
	OD = (200*VP)/((2*VP)+FP+FN)
	OR = (100*VP)/(VP + FP + FN)

	resultado = [VP, FP, FN, OD, OR]
	
	return resultado