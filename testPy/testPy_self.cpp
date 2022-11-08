#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #define Py_LIMITED_API 0x03060000

#include <Python.h>
#include <iostream>
#include <numpy/arrayobject.h>
// Refer to the following website for more information about embedding the
// Python code in C++.
// https://docs.python.org/3/extending/embedding.html
int main(int argc, char* argv[]) {
    PyObject *pName, *pModule, *pClass, *pInstance, *pTuple, *pTuple_empty,*pFunc_init, *pFunc_add;
    PyArrayObject *np_ret, *np_arg_1, *np_arg_2, *np_arg_3;

    // Initializes the Python interpreter
    // wchar_t *pProgramName, *pPathName
    Py_SetPath(L"/mnt/e/Linux_space/VirtualEnv/TF2/lib/python36.zip:"
               "/mnt/e/Linux_space/VirtualEnv/TF2/lib64/python3.6:"
               "/mnt/e/Linux_space/VirtualEnv/TF2/lib64/python3.6/lib-dynload:"
               "/usr/lib64/python3.6:"
               "/usr/lib/python3.6:"
               "/mnt/e/Linux_space/VirtualEnv/TF2/lib/python3.6/site-packages:");

    //

    Py_SetProgramName(L"testAdder");
    // Py_SetPythonHome(L"/mnt/e/Linux_space/VirtualEnv/TF2");

    Py_Initialize();
    // Necessary packages
    // PyRun_SimpleString(
    //     "import sys\n"
    //     "sys.path.insert(0,'')");
    PyRun_SimpleString(
        "import sys");

    PyRun_SimpleString("print(sys.path)");

    pName = PyUnicode_DecodeFSDefault("adder");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    // Load the py_module object
    if (pModule == nullptr) {
        PyErr_Print();
        std::cerr << "Fails to import the pModule.\n";
        return 1;
    }

    pClass = PyObject_GetAttrString(pModule, "adder_class");
    if (pClass == nullptr) {
        PyErr_Print();
        std::cerr << "Fails to find the pClass.\n";
        return 1;
    }

    pTuple_empty = PyTuple_New(0);

    pInstance = PyObject_CallObject(pClass, pTuple_empty);

    if (pInstance == nullptr) {
        PyErr_Print();
        std::cerr << "Fails to find the pInstance.\n";
        return 1;
    }

    pFunc_add = PyObject_GetAttrString(pInstance, "add");

    import_array();
    int n_data = 100;

    npy_intp dims[] = {n_data};
    np_arg_1 = (PyArrayObject*)PyArray_SimpleNew(1, &dims[0], NPY_DOUBLE);
    np_arg_2 = (PyArrayObject*)PyArray_SimpleNew(1, &dims[0], NPY_DOUBLE);
    np_arg_3 = (PyArrayObject*)PyArray_SimpleNew(1, &dims[0], NPY_DOUBLE);

    pTuple = PyTuple_New(2);
    PyTuple_SetItem(pTuple, 0, (PyObject*)np_arg_2);
    PyTuple_SetItem(pTuple, 1, (PyObject*)np_arg_3);
    


    // Execute function

    double* arg_1 = (double*)PyArray_DATA(np_arg_1);
    double* arg_2 = (double*)PyArray_DATA(np_arg_2);
    double* arg_3 = (double*)PyArray_DATA(np_arg_3);

    for (int i = 0; i < n_data; i++) {
        arg_1[i] = i;
        arg_2[i] = 2 * i;
        arg_3[i] = 3 * i;
    }

    PyObject* result = PyObject_CallObject(pFunc_add, pTuple);
    if (result == nullptr) {
        PyErr_Print();
        std::cerr << "Fails to get the result.\n";
        return 1;
    }

    np_ret = (PyArrayObject*)result;
    double* d_out= (double*)PyArray_DATA(np_ret);

    for (int i = 0; i < n_data; i++) {
        std::cout << d_out[i] << ", ";
    }

    std::cout << " ... end" << std::endl;
    Py_Finalize();

    return 0;
}