"""
RECHERCHE OPÉRATIONNELLE - TP Bellman Ford

GROUPE :
- Aigle Dimitri
- Couvin Quetsiah
- Gouptar-Ticket Yanissa
- Kancel Jonathan

"""
import numpy


def bellman_ford(Z, sommet_de_depart):
    ordre = numpy.shape(Z)[0]

    # Initialisation
    d = [1e999] * (ordre + 1)
    d = numpy.full(ordre, 1e999)
    d[sommet_de_depart] = 0

    for i in range(0, len(Z) + 1):
        # Parcours de la matrice
        for index_ligne, i in enumerate(Z):

            # Récupération de la dernière étape
            d_test = d[-1] if d.ndim > 1 else d

            for index_colonne, j in enumerate(i):

                # verifier que le coût ne soit pas null
                if j != 0:
                    d_test[index_colonne] = min(d_test[index_colonne], d_test[index_ligne] + j)
                    # print(d_test)

            if d[-1].all() != d_test.all():
                d = numpy.vstack([d, d_test])

    init = numpy.full(ordre, 1e999)
    init[sommet_de_depart] = 0

    return numpy.vstack([init, d])


if __name__ == '__main__':
    Z = numpy.array([
        [0, 15, 0, 0, 8, 13, 6],
        [15, 0, 10, 13, 0, 0, 14],
        [0, 10, 0, 12, 0, 11, 0],
        [0, 13, 12, 0, 11, 5, 12],
        [8, 0, 0, 11, 0, 5, 10],
        [13, 0, 11, 5, 5, 0, 0],
        [6, 14, 0, 12, 10, 0, 0]
    ])

    t = numpy.array([
        [0, 6, 7, 0, 0],
        [0, 0, 8, 5, -4],
        [0, 0, 0, 3, 9],
        [0, -2, 0, 0, 0],
        [-2, 0, 0, 7, 0]
    ])

    p = numpy.array([
        [0, 3, 4, 0, 0, 0],
        [0, 0, 9, 2, 2, 0],
        [0, 0, 0, -5, 0, 0],
        [0, -2, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])

    result = bellman_ford(Z, 0)

    print(result)