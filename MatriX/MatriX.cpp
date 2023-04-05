#include <iostream>
#include <exception>

template <typename T>
class Allocator
{
public:
    static T** allocate(size_t rows, size_t cols);
    static void deallocate(T** ptr, size_t rows);
};

template <typename T>
T** Allocator<T>::allocate(size_t rows, size_t cols)
{
    T** ptr = new T * [rows];
    try {
        for (size_t i = 0; i < rows; i++)
        {
            ptr[i] = new T[cols];
        }
    }
    catch (...)
    {
        Allocator::deallocate(ptr, rows);
        std::cerr << "dfdf";
        throw std::out_of_range("Index out of range");
    }

    return ptr;
}

template <typename T>
void Allocator<T>::deallocate(T** ptr, size_t rows)
{
    for (size_t i = 0; i < rows; i++)
    {
        delete[] ptr[i];
    }
    delete[] ptr;
    ptr = nullptr;
}


template <typename T>
class AbstractMatrix
{
public:
    virtual size_t getRows() = 0;
    virtual size_t getCols() = 0;
    virtual T operator()(size_t rows, size_t cols) = 0;
    //virtual const T& operator()(size_t rows, size_t cols) const = 0;
    //virtual void resize() = 0;

};

template <typename T>
class Matrix: public AbstractMatrix<T>
{
public:
    Matrix(size_t rows = 1, size_t cols = 1, const T value = 1);
    Matrix(size_t rows, size_t cols, T** arr);
    Matrix(const Matrix<T>& other);
    ~Matrix();


    size_t getRows() override;
    size_t getCols() override;
    T operator()(size_t rows, size_t cols) override;
    Matrix<T>& operator=(Matrix<T>& other);
    Matrix<T>& operator+(Matrix<T>& other);
    Matrix<T>& operator-(Matrix<T>& other);
    Matrix<T>& operator*(Matrix<T>& other);
    Matrix<T>& ortogonal(Matrix<T>& other);
    Matrix<T>& transpose();

    friend std::ostream& operator<<(std::ostream& os, Matrix<T>& matrix)
    {
        for (size_t i = 0; i < matrix.rows_; i++)
        {
            for (size_t j = 0; j < matrix.cols_; j++)
            {
                os << matrix(i,j) << " ";
            }
            os << std::endl;
        }
        return os;
    }

private:
    size_t rows_;
    size_t cols_;
    T** data_;
};

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols,const T value):rows_(rows), cols_(cols), data_(Allocator<T>::allocate(rows_,cols_))
{
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            data_[i][j] = value;
        }
    }
}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, T** arr):rows_(rows), cols_(cols), data_(Allocator<T>::allocate(rows_,cols_))
{

    if (arr == nullptr)
    {
        std::cerr << "Array pointer is null" << std::endl;
        throw std::invalid_argument("Array pointer is null");
    }
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            data_[i][j] = arr[i][j];
        }
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T>& other): rows_(other.rows_), cols_(other.cols_),data_(Allocator<T>::allocate(rows_,cols_))
{
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            data_[i][j] = other.data_[i][j];
        }
    }
}

template <typename T>
Matrix<T>::~Matrix()
{
    Allocator<T>::deallocate(data_, rows_);
    std::cout << "Memory if free" << std::endl;
}

template <typename T>
size_t Matrix<T>::getRows() { return rows_; }

template <typename T>
size_t Matrix<T>::getCols() { return cols_; }

template <typename T>
T Matrix<T>::operator()(size_t rows, size_t cols)
{
    if (rows > rows_ || cols > cols_)
    {
        throw std::out_of_range("Index out of range");
    }

    return data_[rows][cols];
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>& other)
{
    if (this != &other) {
        Allocator<T>::deallocate(data_, rows_);

        rows_ = other.rows_;
        cols_ = other.cols_;

        data_ = Allocator<T>::allocate(rows_, cols_);
        for (size_t i = 0; i < rows_; i++) 
        {
            for (size_t j = 0; j < cols_; j++) 
            {
                data_[i][j] = other.data_[i][j];
            }
        }
    }
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+(Matrix<T>& other)
{
    if(rows_ != other.rows_ || cols_ != other.cols_) 
    {
        std::cerr << "Matrices have different dimensions" << std::endl;
        throw std::invalid_argument("Matrices have different dimensions");
    }

    Matrix<T>* result = new Matrix<T>(rows_, cols_);
    for (size_t i = 0; i < rows_; i++) 
    {
        for (size_t j = 0; j < cols_; j++) 
        {
            result->data_[i][j] = data_[i][j] + other(i, j);
        }
    }
    return *result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-(Matrix<T>& other)
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        std::cerr << "Matrices have different dimensions" << std::endl;
        throw std::invalid_argument("Matrices have different dimensions");
    }

    Matrix<T>* result = new Matrix<T>(rows_, cols_);
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            result->data_[i][j] = data_[i][j] - other(i, j);
        }
    }
    return *result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*(Matrix<T>& other)
{
    if (cols_ != other.rows_) 
    {
        std::cerr << "Matrices have different dimensions" << std::endl;
        throw std::invalid_argument("Matrices are not compatible for multiplication");
    }
    Matrix<T>* result = new Matrix<T>(rows_, other.cols_,0);

    for (int i = 0; i < rows_; i++) 
    {
        for (int j = 0; j < other.cols_; j++) 
        {
            for (int k = 0; k < cols_; k++) 
            {
                result->data_[i][j] += data_[i][k] * other(k, j);
            }
        }
    }

    return *result;
}

template <typename T>
Matrix<T>& Matrix<T>::transpose() 
{
    Matrix<T>* result =  new Matrix<T>(cols_, rows_); // создаем новую матрицу размера m_cols x m_rows
    for (int i = 0; i < rows_; i++) 
    {
        for (int j = 0; j < cols_; j++)
        {
            result->data_[j][i] = (*this)(i, j); 
        }
    }
    return *result;
}

template <typename T>
class Matrix2D :public Matrix<T>
{
public:
    Matrix2D(size_t size, const T value = 1) :Matrix<T>(size, size,value) {}
    Matrix2D(size_t size, T** arr) :Matrix<T>(size, size, arr){}
    using Matrix<T>::operator=;

    double determinant();

    bool isSquare() 
    {
        return (this->getCols() == this->getRows());
    }

};

template <typename T>
double Matrix2D<T>::determinant()
{

}

template <typename T>
class InvertibleMatrix : public Matrix2D<T>
{

};

int main()
{
    int N = 3;
    int** arr = new int* [N];
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = new int[N];
        for (size_t j = 0; j < N; j++)
        {
            arr[i][j] = rand()%20;
        }
    }

    Matrix<int> B(3,3,5);
    Matrix<int> D(3,3);
    Matrix<int> C(N,N,arr);
    B = D;
    B = D - C;
    B = D * C;
    std::cout << D;
    std::cout << C;

    C = C.transpose();

    std::cout << C;

    int a = 5;

    Matrix2D<int> H(a,3);
    Matrix2D<int> M(a);



    return 0;
}
