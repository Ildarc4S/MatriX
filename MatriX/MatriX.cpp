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
        for (size_t i = 0; i < cols; i++)
        {
            ptr[i] = new T[cols];
        }
    }
    catch (...)
    {
        Allocator::deallocate(ptr, rows);
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
}


template <typename T>
class AbstractMatrix
{
public:
    virtual size_t getRows() = 0;
    virtual size_t getCols() = 0;
    virtual T operator()(size_t rows, size_t cols) = 0;
    //virtual const T& operator()(size_t rows, size_t cols) const = 0;

};

template <typename T>
class Matrix: public AbstractMatrix<T>
{
public:
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(T** arr, size_t cols,size_t rows);
    //Matrix(const Matrix<T>& other);


    size_t getRows() override;
    size_t getCols() override;
    T operator()(size_t rows, size_t cols) override;
    //const T& operator()(size_t rows, size_t cols) const override;

private:
    size_t rows_;
    size_t cols_;
    T** data_;
};

template <typename T>
Matrix<T>::Matrix()
{
    rows_ = 0;
    cols_ = 0;
    data_ = nullptr;
}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols):rows_(rows), cols_(cols), data_(Allocator<T>::allocate(rows_,cols_)){}

template <typename T>
Matrix<T>::Matrix(T** arr, size_t cols, size_t rows):rows_(rows), cols_(cols), data_(Allocator<T>::allocate(rows_,cols_))
{
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            data_[i][j] = arr[i][j];
        }
    }
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


int main()
{
    Matrix<int> A(1, 2);
    Matrix<float> B(3, 3);


    return 0;
}
