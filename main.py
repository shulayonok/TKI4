import numpy as np


class AdvField:
    def __init__(self, p, r, fx):
        self.p = p
        self.r = r
        self.q = p ** r
        self.fx_coef = fx
        self.fx = np.poly1d(fx)
        self.prim = np.poly1d(np.mod((-1 * self.fx.coef[1:]), p))
        self.get_elements()  # Словарь степеней и векторов поля
        self.get_zech()  # Таблица Логарифмов

    # Вывод примитивного элемента, многочлена
    def show_pirimitive(self):
        print(f'Примитивный элемент: \n{self.prim}')

    # Вычисление всех элементов поля
    def get_elements(self):
        self.steps = {}
        for i in range(self.r):
            vec = np.zeros(self.r, dtype=np.int32).tolist()
            vec[self.r - i - 1] = 1
            self.steps[i] = vec
        self.steps[self.r] = [0] * (self.r - self.prim.coef.shape[0]) + self.prim.coef.tolist()
        for i in range(self.r + 1, self.q - 1):
            cur_p = np.mod(np.polydiv(np.polymul(self.steps[i - 1], self.steps[1]), self.fx)[-1], self.p)
            self.steps[i] = [0] * (self.r - cur_p.shape[0]) + cur_p.astype(np.int32).tolist()
        self.vectors = {str(v): k for k, v in self.steps.items()}

    # Нахождение таблицы логарифмов
    def get_zech(self):
        self.logtable = {0: '-'}
        for i in range(1, self.q - 1):
            cur = np.mod(np.polyadd(self.steps[i], [1]), self.p).astype(np.int32).tolist()
            for k, v in self.steps.items():
                if v == cur:
                    self.logtable[i] = k
                    break

    # Вывести элемент степени i
    def get_element(self, i):
        idx = i % (self.q - 1)
        return self.steps[idx]

    # Сложить два элемента поля
    def add(self, i, j):
        if i > j:
            i, j = j, i
        z_i = (j - i) % (self.q - 1)
        if self.logtable[z_i] == '-':
            return [0]
        return self.steps[(i + self.logtable[z_i]) % (self.q - 1)]

    # Умножить два элемента поля
    def mult(self, i, j):
        return self.steps[(i + j) % (self.q - 1)]


class ElementGF:
    def __init__(self, step, gf):
        self.step = step
        self.gf = gf
        self.q = gf.q
        self.logtable = gf.logtable

    # Вывести элемент поля в виде степени
    def __str__(self):
        if self.step == '-':
            return "0"
        indexes = {"0": "0",
                   "1": "1",
                   "2": "2",
                   "3": "3",
                   "4": "4",
                   "5": "5",
                   "6": "6",
                   "7": "7",
                   "8": "8",
                   "9": "9",
                   "-": "\u207B"}
        degrees, temp = "^", str(self.step)
        for char in temp:
            degrees += indexes[char] or ""
        return 'a' + degrees

    # Метод возвращает элемент поля ввиде степени
    def degree(self):
        return self.step

    # Метод возвращает элемент поля ввиде вектора
    def vector(self):
        if self.step == '-':
            return [0] * self.gf.r
        return self.gf.get_element(self.step)

    # Перегрузка оператора сложения
    def __add__(self, other):
        i = self.step
        j = other.step
        if i == '-':  # 0 + Alp^j = Alp^j
            return other
        if j == '-':  # Alp^i + 0 = Alp^i
            return self
        if i > j:
            i, j = j, i
        z_i = (j - i) % (self.q - 1)
        if self.logtable[z_i] == '-':
            return ElementGF('-', self.gf)
        obj = ElementGF((i + self.logtable[z_i]) % (self.q - 1), self.gf)
        return obj

    # Перегрузка оператора умножения
    def __mul__(self, other):
        i = self.step
        j = other.step
        if i == '-' or j == '-':
            return ElementGF('-', self.gf)
        obj = ElementGF((i + j) % (self.q - 1), self.gf)
        return obj

    # Перегрузка операции деления
    def __truediv__(self, other):
        i = self.step
        j = other.step
        if i == '-' or j == '-':
            return self
        return ElementGF((i - j) % (self.gf.q - 1), self.gf)

    # Перегрузка оператора возведения в степень
    def __pow__(self, num):
        if self.step == '-':
            return self
        return ElementGF(self.step * num, self.gf)


class PolyGF:
    def __init__(self, coef, gf):
        self.coef = coef  # Коэффиценты многочлена - элементы поля Галуа ввиде степени
        self.deg = len(coef) - 1
        self.gf = gf

    # Степень преобразоать в текстовый вид
    def deg2str(self, a, step):
        indexes = {"0": "0",
                   "1": "1",
                   "2": "2",
                   "3": "3",
                   "4": "4",
                   "5": "5",
                   "6": "6",
                   "7": "7",
                   "8": "8",
                   "9": "9",
                   "-": "\u207B"}
        degrees, temp = "^", str(step)
        for char in temp:
            degrees += indexes[char] or ""
        return a + degrees

    # Метод возвращает копию объекта
    def copy(self):
        return PolyGF(self.coef.copy(), self.gf)

    # Перегрузка оператора строкового представления
    def __str__(self):
        text = ''

        for k, c in enumerate(self.coef[:-1]):
            if c != '-':
                text += str(ElementGF(c, self.gf)) + self.deg2str('x', self.deg - k) + '+'
        if self.coef[-1] != '-':
            text += str(ElementGF(self.coef[-1], self.gf))
        else:
            text = text[:-1]
        return text

    # Удаляет нулевые (избыточные) коэффиценты слева
    def clear_inf(self):
        for i in range(len(self.coef) - 1):
            if self.coef[0] == '-':
                self.coef.pop(0)
            else:
                break
        self.deg = len(self.coef) - 1

    # Нахождение знчения многочлена
    def value(self, x):
        v = ElementGF('-', self.gf)
        for i in range(1, len(self.coef) + 1):
            v = v + (ElementGF(self.coef[-i], self.gf)) * (x ** (i - 1))
        return v

    # Вычисление производной многочлена в поле Галуа
    def derivative(self):
        deriv_coef = []
        for i in range(self.deg):
            if (self.deg - i) % self.gf.p == 0:
                deriv_coef.append('-')
            else:
                deriv_coef.append(self.coef[i])
        d = PolyGF(deriv_coef, self.gf)
        d.clear_inf()
        return d

    # Перегрузка операторов сложения полиномов
    def __add__(self, other):
        res = other.copy() if other.deg > self.deg else self.copy()
        for i in range(1, min(other.deg, self.deg) + 2):
            res.coef[-i] = (ElementGF(self.coef[-i], self.gf) + ElementGF(other.coef[-i], other.gf)).step
        res.clear_inf()
        return res

    # Перегрузка операторов умножение полиномов
    def __mul__(self, other):
        p = self.coef
        q = other.coef
        res = PolyGF(['-'] * (len(p) + len(q) - 1), self.gf)
        for i, ci in enumerate(p):
            for j, cj in enumerate(q):
                temp = ElementGF(ci, self.gf) * ElementGF(cj, self.gf)
                res.coef[i + j] = (ElementGF(res.coef[i + j], self.gf) + temp).step
        res.clear_inf()
        return res

    # Перегрузка операции деления
    def __truediv__(self, other):
        if other.deg > self.deg:
            return self
        p = self.copy()
        q = other.copy()
        if q.deg == 0:
            c = q.coef[-1]
            if c == '-':
                return self
            else:
                new_coef = ['-' if p_i == '-' else (p_i - c) % (self.gf.q - 1) for p_i in p.coef]
                return PolyGF(new_coef, self.gf)
        res = []
        while p.deg + 1 > q.deg:
            t = (p.coef[0] - q.coef[0]) % (self.gf.q - 1)
            deg = p.deg - q.deg
            res.append(t)
            t = q * PolyGF([t] + ['-'] * deg, self.gf)
            p = p + t
        return PolyGF(res, self.gf)

    # Перегрузка операции деления со взятием остатка
    def __mod__(self, other):
        if other.deg > self.deg:
            return self
        p = self.copy()
        q = other.copy()
        while p.deg + 1 > q.deg:
            t = (p.coef[0] - q.coef[0]) % (self.gf.q - 1)
            deg = p.deg - q.deg
            t = q * PolyGF([t] + ['-'] * deg, self.gf)
            p = p + t
        return p


class RS:
    def __init__(self, d, k, GF_tool, shorted=0):
        self.k = k + shorted
        self.d = d + 1
        self.n = self.d + self.k - 1
        self.t = (self.d - 1) // 2
        self.gf = GF_tool
        self.shorted = shorted
        self.init()

    # Метод находит порождающий/проверочный многочлен/матрицу
    def init(self):
        # Вычисление порождающего многочлена
        self.gx = PolyGF([0], self.gf)
        for i in range(1, self.d):
            self.gx = self.gx * PolyGF([0, i], self.gf)
        # Вычисление проверочного многочлена
        self.hx = PolyGF([0], self.gf)
        for i in range(self.d, self.n + 1):
            self.hx = self.hx * PolyGF([0, i], self.gf)
        # Вычисление порождающий матрицы
        self.G = []
        for i, j in enumerate(range(self.n - 1, self.d - 2, -1)):
            vec = ['-'] * self.k
            vec[i] = 0
            vec += (PolyGF([0] + ['-'] * j, self.gf) % self.gx).coef
            self.G.append(vec)
        self.G = list(map(list, zip(*self.G)))
        # Вычисление проверочной матрицы
        self.H = []
        for i in range(1, self.d):
            self.H.append((np.arange(self.gf.q - 1) * i).tolist())

    # Вычисление скалярного произведения из элеметов поля Галуа
    def scalar(self, a, b):
        value = ElementGF('-', self.gf)
        length = min(a.deg, b.deg)
        for i in range(length + 1):
            value = value + (ElementGF(a.coef[i], self.gf) * ElementGF(b.coef[i], self.gf)
                             )
        return value

    # Вычисление синдромов
    def syndrome(self, message):
        # Добавляем нули для укороченного кода
        m_coef = message.coef
        message = PolyGF(m_coef[:self.k - self.shorted] + ['-'] * self.shorted + m_coef[self.k - self.shorted:],
                         self.gf)
        res = []
        for i in range(2 * self.t):  # Умножение Матрицы H на вектор Кода
            res.append(self.scalar(PolyGF(self.H[i][::-1], self.gf), message).step)
        return res

    # Алгоритм Берлекэмпа-Месси
    def find_locator(self, S, verbose=False):
        k = 0
        Cx = PolyGF([0], self.gf)  # Λ(x)
        L = 0
        Bx = PolyGF([0], self.gf)  # Вспомогательный многочлен
        delta, Tx = None, None  # Δ, T(x)
        if verbose:
            print('Таблица алгоритма Берлекэмпа-Месси')
            print('k | Δ | Λ(x) | L | T(x)')
        while k < self.t * 2:
            if verbose:
                print('{} | {} | {} | {} | {}'.format(k, delta, Cx, L, Tx))
            k += 1
            delta = ElementGF('-', self.gf)
            for j in range(min(L + 1, len(Cx.coef))):
                delta = delta + (ElementGF(Cx.coef[-j - 1], self.gf) * ElementGF(S[k - j - 1], self.gf))
            if delta.step == '-':
                Bx = PolyGF([0, '-'], self.gf) * Bx
                continue
            Tx = Cx + PolyGF([delta.step], self.gf) * PolyGF([0, '-'], self.gf) * Bx
            if 2 * L > k - 1:
                Cx = Tx.copy()
                Bx = PolyGF([0, '-'], self.gf) * Bx
                continue
            Bx = Cx / PolyGF([delta.step], self.gf)
            Cx = Tx.copy()
            L = k - L
        if verbose:
            print('{} | {} | {} | {} | {}'.format(k, delta, Cx, L, Tx))
        return Cx

    # Поиск мест ошибок
    def find_position_error(self, Lx):
        pos = []
        for i in range(self.n):
            if str(Lx.value(ElementGF(i, self.gf))) == '0':
                pos.append((-i - self.shorted) % (self.gf.q - 1))
        return pos

    # Поиск величин ошибок
    def find_value_error(self, pos, Sx, Lx, verbose=False):
        Sigma = (Sx * Lx) % PolyGF([0] + ['-'] * 2 * self.t, self.gf)
        Lx_der = Lx.derivative()
        if verbose:
            print('Ω(x) =', str(Sigma))
            print('Λ\'(x) =', str(Lx_der))
        val = []
        for i in pos:
            p = self.gf.q - 1 - i - self.shorted
            v = (Sigma.value(ElementGF(p, self.gf)) / Lx_der.value(ElementGF(p, self.gf))).step
            val.append(v)
        return val

    # Кодирование
    def encode(self, message):
        # Добавляем нули для укороченного кода
        message = message * PolyGF([0] + ['-'] * self.shorted, self.gf)
        res = []
        for i in range(self.n):  # Умножение Матрицы G на вектор слова
            res.append(self.scalar(PolyGF(self.G[i], self.gf), message).step)
        # Удаляем добавленные нули укороченного кода
        res = res[:self.k - self.shorted] + res[self.k:]
        return PolyGF(res, self.gf)

    # Декодирование
    def decode(self, message):
        S = self.syndrome(message)
        # Нет ошибок
        if S == (['-'] * (self.d - 1)):
            print('Ошибок нет!')
            return PolyGF('-', self.gf)
        # Есть ошибки
        Lx = self.find_locator(S)
        Pos = self.find_position_error(Lx)
        print('Ошибки на позициях:', Pos)
        Sx = PolyGF(S.copy()[::-1], GF)
        Val = self.find_value_error(Pos, Sx, Lx)
        print('Величины ошибок:', [str(ElementGF(v, self.gf)) for v in Val])
        # Многочлен ошибки
        E = ['-'] * self.n
        for i, v in zip(Pos, Val):
            E[i] = v
        E.reverse()
        E = PolyGF(E, self.gf)
        E.clear_inf()
        return E


n = 100
k = 96
d = 4

shorted = 27

r = 7
q = 2 ** r
fx = np.poly1d([1, 0, 0, 0, 1, 0, 0, 1])
# 1
# a
print("a")
GF = AdvField(p=2, r=r, fx=[1, 0, 0, 0, 1, 0, 0, 1])
GF.show_pirimitive()

print('Элементы поля:')
for key, val in GF.steps.items():
    print(str(ElementGF(key, GF)) + ' - ' + str(val))

print("\nb")

# b
# Создаем объект класса Кода Рида-Соломона с укорочением
codeRS = RS(d, k, GF, shorted=shorted)

# Порождающий многочлен
print()
print(codeRS.gx)
print("\nc")

# c
# Элементы в виде степени (Alp^(-) = 0)
# Порождающая матрица
print()
print(np.array(codeRS.G))

# Элементы в виде степени
# Проверочная матрица
print()
print(np.array(codeRS.H))

# 2
# Информационный многочлен
print("\n2")
print()
M = PolyGF(np.random.randint(0, q - 1, k).tolist(), GF)
print(M)

# Кодовое слово
print()
C = codeRS.encode(M)
print(C)

# 3
# Полином из двух ошибок
print("\n3")
e = ['-'] * (n)
e[55] = 35  # Alpha^35 * X^55
e[19] = 12  # Alpha^12 * X^19
E = PolyGF(e[::-1], GF)
print()
print('Многочлен ошибки:', E)

# Добавляем ошибки
V = C + E
print()
print('Многочлен с ошибкой:')
print(V)
print()
print('Синдромы:')
S = codeRS.syndrome(V)
for i, s in enumerate(S):
    print('S{} = {}'.format(i + 1, str(ElementGF(s, GF))))

# 4
print("\n4")
Lx = codeRS.find_locator(S, verbose=True)
print()
print('Многочлен локатор ошибок:', Lx)

pos = np.array(codeRS.find_position_error(Lx))
print()
print('Позиции ошибок:', pos)

# 5
print("\n5")
Sx = PolyGF(S.copy()[::-1], GF)
# Величина ошибки в виде степени
# Используется алгоритм Форни
alp_value = codeRS.find_value_error(pos, Sx, Lx, verbose=True)
print()
print('Значение ошибок:', alp_value)

# Итоговое декодирование
print()
print(codeRS.decode(V))
