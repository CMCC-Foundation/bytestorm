DENSITY_MAP_TC = 'density_map_tc'
SQUARE_MAP_TC = 'square_map_tc'
LABEL_MAP_TC = 'label_map_tc'

BASINS_COORDS = {
    'NI': ((0,70),(20,100)),
    'WNP': ((0,70),(100,180)),
    'WP': ((0,70),(100,180)),
    # ENP e NATL si sovrappongono a causa del Nord America
    'ENP': ((0,70),(180,320)),
    'EP': ((0,70),(180,320)),
    'NATL': ((0,70),(280,360)),
    # Allungato di 30° ad Ovest perché ci sono
    # alcuni TC con LON che arriva fino a 252.25°
    'NA': ((0,70),(280-30,360)),
    # Allungato di 5 gradi per arrotondare sulle 
    # patch, pertanto si sovrappone al bacino SP
    # Allungato di 10° ad Ovest perché ci sono 
    # alcuni TC con LON che arriva fino a 11°.
    'SI': ((-70,0),(20-10,135+5)),
    'SP': ((-70,0),(135,315)),
    # SATL è un po più piccolo, dovrebbe arrivare a 20E
    # Tuttavia, non è necessario aggiungere quel pezzo
    # perchè ci sono pochissimi TC in quel bacino
    # (solamente 61 TC detection in 40 anni)
    # Inoltre, prende una piccolissima parte del SP
    'SATL': ((-70,0),(315-5,360)),
    'SA': ((-70,0),(315-5,360)),
}
