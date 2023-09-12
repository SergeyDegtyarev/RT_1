import numpy as np

# закон отражения в векторной форме, возвращает единичный вектор направления отраженного луча
def law_refl(direction, normal):
    refl = direction - 2 * direction.dot(normal) * normal
    return refl


# закон преломления в векторной форме, возвращает единичный вектор направления преломленного луча
def law_refr(direction, normal, ind1, ind2):
    if ((ind2 ** 2 - ind1 ** 2) / ((ind1 * direction).dot(normal)) ** 2 + 1) >= 0:
        refr = ind1 * direction - (ind1 * direction).dot(normal) * normal * (1 - np.sqrt((ind2 ** 2 - ind1 ** 2) / ((ind1 * direction).dot(normal)) ** 2 + 1))
    else:
        return law_refl(direction, normal)
    return np.divide(refr, ind2)