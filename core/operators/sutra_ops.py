from fractions import Fraction

from vedic_book import SUTRAS
from vedic_field import FieldBinding

from .base import Operator, OperatorCategory, OperatorContext, CompositeOperator
from ..state import FieldState, RationalComplex


class SutraOperator(Operator):
    sutra_id = None

    def __init__(self):
        if self.sutra_id is None:
            raise TypeError("Use a specific book sutra operator")
        record = SUTRAS[self.sutra_id - 1]
        super().__init__(name=record.name, category=OperatorCategory.FIELD)
        self.sutra_number = record.id
        self.sanskrit_name = record.name

    def binding(self, context):
        operands = context.get_param("sutra_operands", {})
        if self.sutra_number in operands and str(self.sutra_number) in operands:
            raise ValueError("Duplicate integer and string sutra operand keys")
        parameters = operands.get(self.sutra_number, operands.get(str(self.sutra_number)))
        return FieldBinding.from_parameters(self.sutra_number, parameters)

    def metadata(self, context):
        return self.binding(context).metadata("matched_edge_rotations")

    def apply(self, state, context):
        return self.apply_parallel(state, context)

    def apply_parallel(self, state, context, executor=None, chunk_size=256):
        binding = self.binding(context)
        points = tuple(state.lattice.iterate_all())
        indices = {point.coords: i for i, point in enumerate(points)}
        neighbors = tuple(tuple(indices[neighbor.coords]
                                for neighbor in state.lattice.nearest_neighbors(point))
                          for point in points)
        values = tuple(state.get(point) for point in points)
        strength = context.get_param("sutra_strength", context.dt)
        if executor is None:
            result = binding.rotate_matched(values, neighbors, strength)
        else:
            real_job = executor.submit(binding.rotate_matched, tuple(v.real for v in values), neighbors, strength)
            imag_job = executor.submit(binding.rotate_matched, tuple(v.imag for v in values), neighbors, strength)
            result = tuple(RationalComplex(real, imag) for real, imag in zip(real_job.result(), imag_job.result()))
        context.set_param("sutra_execution",
                          binding.receipt(values, result, neighbors, strength, "matched_edge_rotations"))
        output = state.copy()
        for point, value in zip(points, result):
            output.set(point, value)
        return output

    def sutra_transform(self, value, coords, state, context):
        if value != state.get_by_coords(*coords):
            raise ValueError("The point value does not match the supplied field state")
        return self.apply(state, context).get_by_coords(*coords)


class Sutra01_EkadhikenaPurvena(SutraOperator):
    sutra_id = 1


class Sutra02_NikhilamNavatashcaramam(SutraOperator):
    sutra_id = 2


class Sutra03_UrdhvaTiryagbhyam(SutraOperator):
    sutra_id = 3


class Sutra04_ParavartyaYojayet(SutraOperator):
    sutra_id = 4


class Sutra05_ShunyamSamuccaye(SutraOperator):
    sutra_id = 5


class Sutra06_AnurupyeSunyamanyat(SutraOperator):
    sutra_id = 6


class Sutra07_SankalanVyavakalanabhyam(SutraOperator):
    sutra_id = 7


class Sutra08_Puranapuranabhyam(SutraOperator):
    sutra_id = 8


class Sutra09_CalanaKalanabhyam(SutraOperator):
    sutra_id = 9


class Sutra10_Yavadunam(SutraOperator):
    sutra_id = 10


class Sutra11_Vyashtisamashti(SutraOperator):
    sutra_id = 11


class Sutra12_Sesanyankena(SutraOperator):
    sutra_id = 12


class Sutra13_Sopantyadvayamantyam(SutraOperator):
    sutra_id = 13


class Sutra14_EkanyunenaPurvena(SutraOperator):
    sutra_id = 14


class Sutra15_Gunitasamuccayah(SutraOperator):
    sutra_id = 15


class Sutra16_Gunakasamuccayah(SutraOperator):
    sutra_id = 16


class SubSutra17_Anurupyena(SutraOperator):
    sutra_id = 17


class SubSutra18_SisyateSesasamjnah(SutraOperator):
    sutra_id = 18


class SubSutra19_Adyamadyenantyamantyena(SutraOperator):
    sutra_id = 19


class SubSutra20_KevalaihSaptakam(SutraOperator):
    sutra_id = 20


class SubSutra21_Vestanam(SutraOperator):
    sutra_id = 21


class SubSutra22_YavadunamTavadunam(SutraOperator):
    sutra_id = 22


class SubSutra23_YavadunamTavadunikrtya(SutraOperator):
    sutra_id = 23


class SubSutra24_Antyayordasakepi(SutraOperator):
    sutra_id = 24


class SubSutra25_Antyayoreva(SutraOperator):
    sutra_id = 25


class SubSutra26_Samuccayagunitah(SutraOperator):
    sutra_id = 26


class SubSutra27_Lopanasthapanabhyam(SutraOperator):
    sutra_id = 27


class SubSutra28_Vilokanam(SutraOperator):
    sutra_id = 28


class SubSutra29_GunitasamuccayahSamuccayagunitah(SutraOperator):
    sutra_id = 29


Sutra06_Anurupyena = Sutra06_AnurupyeSunyamanyat
Sutra11_Vyashtisamanstih = Sutra11_Vyashtisamashti
Sutra12_Shesanyankena = Sutra12_Sesanyankena
SubSutra17_AnurupyenaSunyamanyat = SubSutra17_Anurupyena
SubSutra18_Yavadunam = SubSutra18_SisyateSesasamjnah
SubSutra20_KevalaiSaptakam = SubSutra20_KevalaihSaptakam
SubSutra21_Veshtanam = SubSutra21_Vestanam
SubSutra22_YavadumamTavadum = SubSutra22_YavadunamTavadunam
SubSutra23_AntyayorDashakepi = SubSutra24_Antyayordasakepi
SubSutra24_AntyayorEva = SubSutra25_Antyayoreva
SubSutra25_Samuccayagunitah = SubSutra26_Samuccayagunitah
SubSutra26_LopanaSthapanabhyam = SubSutra27_Lopanasthapanabhyam
SubSutra27_Vilokanam = SubSutra28_Vilokanam
SubSutra28_GunitasamuccayahSamuccayagunitah = SubSutra29_GunitasamuccayahSamuccayagunitah


SUTRA_CLASSES = (
    Sutra01_EkadhikenaPurvena,
    Sutra02_NikhilamNavatashcaramam,
    Sutra03_UrdhvaTiryagbhyam,
    Sutra04_ParavartyaYojayet,
    Sutra05_ShunyamSamuccaye,
    Sutra06_AnurupyeSunyamanyat,
    Sutra07_SankalanVyavakalanabhyam,
    Sutra08_Puranapuranabhyam,
    Sutra09_CalanaKalanabhyam,
    Sutra10_Yavadunam,
    Sutra11_Vyashtisamashti,
    Sutra12_Sesanyankena,
    Sutra13_Sopantyadvayamantyam,
    Sutra14_EkanyunenaPurvena,
    Sutra15_Gunitasamuccayah,
    Sutra16_Gunakasamuccayah,
    SubSutra17_Anurupyena,
    SubSutra18_SisyateSesasamjnah,
    SubSutra19_Adyamadyenantyamantyena,
    SubSutra20_KevalaihSaptakam,
    SubSutra21_Vestanam,
    SubSutra22_YavadunamTavadunam,
    SubSutra23_YavadunamTavadunikrtya,
    SubSutra24_Antyayordasakepi,
    SubSutra25_Antyayoreva,
    SubSutra26_Samuccayagunitah,
    SubSutra27_Lopanasthapanabhyam,
    SubSutra28_Vilokanam,
    SubSutra29_GunitasamuccayahSamuccayagunitah,
)


def get_all_sutras():
    return [operator() for operator in SUTRA_CLASSES]


def get_sutra_by_number(number):
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= 29:
        raise ValueError("The sutra number must be an integer between one and twenty-nine")
    return SUTRA_CLASSES[number - 1]()


def get_sutras_by_category(category):
    return [operator for operator in get_all_sutras() if operator.category == category]


def create_sutra_pipeline(sutra_numbers):
    return CompositeOperator([get_sutra_by_number(number) for number in sutra_numbers])


def _self_test():
    from ..lattice import create_3d_lattice
    from ..state import create_zero_field
    lattice = create_3d_lattice(2, 2, 2)
    state = create_zero_field(lattice)
    state.set_by_coords((0, 0, 0), RationalComplex.one())
    for operator in get_all_sutras():
        result = operator(state, OperatorContext())
        if not operator.check_invariants(result, OperatorContext())[1]:
            raise ArithmeticError(f"S{operator.sutra_number}: field invariants failed")
    return True


if __name__ == "__main__":
    _self_test()
