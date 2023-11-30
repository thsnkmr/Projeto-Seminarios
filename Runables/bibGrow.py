def regionGrow(imagem, objeto, Cmin, Lmin, Cmax, Lmax, Me, Ne, vardp):
    import numpy as np

    io = imagem
    obj = objeto    
    M = Me    
    N = Ne
    
    cmin = Cmin
    lmin = Lmin
    cmax = Cmax
    lmax = Lmax

    seed_lin = int((lmin + lmax)/2)
    seed_col = int((cmin + cmax)/2)

    lst_x = []
    lst_y = []

    lst_x.append(seed_col)
    lst_y.append(seed_lin)

    media = np.mean(io[lmin:lmax,cmin:cmax])
    dp = np.std(io[lmin:lmax,cmin:cmax])

    A = vardp
    limSup = media+A*dp
    limInf = media-A*dp

    obj[seed_lin, seed_col] = 1
    
    (TamanhoFila,) = np.shape(lst_y)
    while TamanhoFila>0:
    	seed_col= lst_x[0]
    	seed_lin = lst_y[0]
    	
    	while (seed_col+1>(N-1)) or (seed_col-1<0) or (seed_lin+1>(M+1)) or (seed_lin-1<0):
    		lst_x.remove(seed_col)
    		lst_y.remove(seed_lin)

    		(TamanhoFila,) = np.shape(lst_y)
    		if TamanhoFila==0:
    			break
    		else:
    			seed_col = lst_x[0]
    			seed_lin = lst_y[0]
                
    	if (obj[seed_lin,seed_col+1]==0) and (io[seed_lin,seed_col+1]>limInf) and (io[seed_lin,seed_col+1]<limSup):
    		lst_x.append(seed_col+1)
    		lst_y.append(seed_lin)

    		obj[seed_lin,seed_col+1] = 1
            
    	if (obj[seed_lin+1,seed_col]==0) and (io[seed_lin+1,seed_col]>limInf) and (io[seed_lin+1,seed_col]<limSup):
    		lst_x.append(seed_col)
    		lst_y.append(seed_lin+1)
    		
    		obj[seed_lin+1,seed_col] = 1

    	if (obj[seed_lin,seed_col-1]==0) and (io[seed_lin,seed_col-1]>limInf) and (io[seed_lin,seed_col-1]<limSup):
    		lst_x.append(seed_col-1)
    		lst_y.append(seed_lin)

    		obj[seed_lin,seed_col-1] = 1
            
    	if (obj[seed_lin-1,seed_col]==0) and (io[seed_lin-1,seed_col]>limInf) and (io[seed_lin-1,seed_col]<limSup):
    		lst_x.append(seed_col)
    		lst_y.append(seed_lin-1)
    		
    		obj[seed_lin-1,seed_col] = 1
            
    	if TamanhoFila == 0:
    		break
    	else:
    		lst_x.remove(seed_col)
    		lst_y.remove(seed_lin)
    		(TamanhoFila,) = np.shape(lst_y)
    
    return(obj)