import numpy as np

one_dimensional_arr = np.array([10, 12])
print(one_dimensional_arr)

a = np.array([1, 2, 3])
print(a)

b = np.arange(3)
print(b)

c = np.arange(1, 20, 3)
print(c)

lin_spaced_arr = np.linspace(0, 100, 5)
print(lin_spaced_arr)

lin_spaced_arr_int = np.linspace(0, 100, 5, dtype=int)
print(lin_spaced_arr_int)

c_int = np.arange(1, 20, 3, dtype=int)
print(c_int)

b_float = np.arange(3, dtype=float)
print(b_float)

char_arr = np.array(['Welcome to Math for ML!'])
print(char_arr)
print(char_arr.dtype)

ones_arr = np.ones(3)
print(ones_arr)

zeros_arr = np.zeros(3)
print(zeros_arr)

rand_arr = np.random.rand(3)
print(rand_arr)

two_dim_arr = np.array([[1,2,3], [4,5,6]])
print(two_dim_arr)


one_dim_arr = np.array([1, 2, 3, 4, 5, 6])

multi_dim_arr = np.reshape(one_dim_arr,(2,3))
print(multi_dim_arr)


arr_1 = np.array([2, 4, 6])
arr_2 = np.array([1, 3, 5])

addition = arr_1 + arr_2
print(addition)

subtraction = arr_1 - arr_2
print(subtraction)

multiplication = arr_1 * arr_2
print(multiplication)


two_dim = np.array(([1, 2, 3],[4, 5, 6],[7, 8, 9]))

print(two_dim[2][1])

a1 = np.array([[1,1],[2,2]])
a2 = np.array([[3,3],[4,4]])
print(f'a1:\n{a1}')
print(f'a2:\n{a2}')

vert_stack = np.vstack((a1, a2))
print(vert_stack)

horz_stack = np.hstack((a1, a2))
print(horz_stack)

A = np.array([[-1, 3],[3, 2]], dtype=np.dtype(float))

b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

x = np.linalg.solve(A, b)
print(f"Solution: {x}")

d = np.linalg.det(A)

print(f"Determinant of matrix A: {d:.2f}")

A_system = np.hstack((A, b.reshape((2, 1))))

print(A_system)

A_system_res = A_system.copy()
A_system_res[1] = 3 * A_system_res[0] + A_system_res[1]

print(A_system_res)

A_2 = np.array([[-1, 3],[3, -9]], dtype=np.dtype(float))

b_2 = np.array([7, 1], dtype=np.dtype(float))

d_2 = np.linalg.det(A_2)

print(f"Determinant of matrix A_2: {d_2:.2f}")

A_2_system = np.hstack((A_2, b_2.reshape((2, 1))))
print(A_2_system)

b_3 = np.array([7, -21], dtype=np.dtype(float))

A_3_system = np.hstack((A_2, b_3.reshape((2, 1))))
print(A_3_system)

A_3_system_res = A_3_system.copy()

A_3_system_res[1] = 3 * A_3_system_res[0] + A_3_system_res[1]
print(A_3_system_res)
