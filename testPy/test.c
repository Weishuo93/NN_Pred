#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

// compiling:
// gcc test.c -o test `pkg-config --cflags --libs python3`

int main() {
    // load python module using C API
    Py_Initialize();

    PyObject* sys = PyImport_ImportModule("sys");
    PyObject* sys_path = PyObject_GetAttrString(sys, "path");
    char * curr = realpath(".", NULL);
    PyObject* curr_str = PyUnicode_FromString(curr);
    free(curr);
    PyList_Append(sys_path, curr_str);
    Py_DECREF(curr_str);
    Py_DECREF(sys_path);
    Py_DECREF(sys);

    PyObject *pModule = PyImport_ImportModule("ramb");
    if (pModule == NULL) {
        fprintf(stderr, "Failed to load %s\n", "ramb");
        Py_Finalize();
        return 1;
    }

    // create instance of class Adder
    PyObject *pClass = PyObject_GetAttrString(pModule, "Adder");
    PyObject *pInstance = PyObject_CallObject(pClass, NULL);

    {
        // 使用类对象的方法，传入三个参数
        PyObject *pFunc = PyObject_GetAttrString(pClass, "add");
        printf("class pFunc: %p\n", pFunc);

        PyObject *pArgs = PyTuple_New(3);
        PyTuple_SetItem(pArgs, 0, pInstance);
        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(1));
        PyTuple_SetItem(pArgs, 2, PyLong_FromLong(2));

        PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
        PyObject_Print(pValue, stderr, 0);
        fprintf(stderr, "\n");

        // Py_XDECREF(pValue);
        // Py_XDECREF(pArgs);
        // Py_XDECREF(pFunc);
    }

    {
        // 使用实例方法，传入两个参数
        PyObject *pFunc = PyObject_GetAttrString(pInstance, "add");
        printf("instance pFunc: %p\n", pFunc);

        PyObject *pArgs = PyTuple_New(2);
        PyTuple_SetItem(pArgs, 0, PyLong_FromLong(1));
        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(2));

        PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
        PyObject_Print(pValue, stderr, 0);
        fprintf(stderr, "\n");

        Py_XDECREF(pValue);
        Py_XDECREF(pArgs);
        // Py_XDECREF(pFunc);
    }

    Py_DECREF(pInstance);
    Py_DECREF(pClass);

    Py_Finalize();
    return 0;
}

