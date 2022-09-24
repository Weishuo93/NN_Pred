/*---------------------------------------------------------------------------*\
Copyright (C) 2022 by Weishuo Liu

License
    This file is part of TF_OF_Predictor

    TF_OF_Predictor is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    TF_OF_Predictor is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with TF_OF_Predictor. If not, see <http://www.gnu.org/licenses/>.

Application
    TF_OF_testAplutB

Description
    Test program of TF_OF_Predictor library to deploy neural network models
    using OpenFOAM "volScalarField" or "scalarField" as input and output

    
\*---------------------------------------------------------------------------*/

#include "scalarField.H"
#include "TF_OF_Predictor.H"

using namespace Foam;

int main() {

    Info << "Preparing raw data ..." << endl
         << endl;

    // create input field a 
    scalarField input_field_a_col1(3, scalar(0.0));
    input_field_a_col1[0] = 0.0;
    input_field_a_col1[1] = 2.2;
    input_field_a_col1[2] = 4.4;
    scalarField input_field_a_col2(3, scalar(0.0));
    input_field_a_col2[0] = 1.1;
    input_field_a_col2[1] = 3.3;
    input_field_a_col2[2] = 5.5;
    // create input field b
    scalarField input_field_b_col1(3, scalar(0.0));
    input_field_b_col1[0] = 5.0;
    input_field_b_col1[1] = 3.0;
    input_field_b_col1[2] = 1.0;
    scalarField input_field_b_col2(3, scalar(0.0));
    input_field_b_col2[0] = 4.0;
    input_field_b_col2[1] = 2.0;
    input_field_b_col2[2] = 0.0;

    // create output field a + b
    scalarField output_field_c_col1(3, scalar(0.0)); // output field for pb model column1
    scalarField output_field_c_col2(3, scalar(0.0)); // output field for pb model column2

    Info << "Creating Data Container\n" << endl;


    // PB format model:
    Foam::List<Foam::List<scalarField*>> multi_inputs(2);
    Foam::List<Foam::List<scalarField*>> multi_outputs(1);

    // For input_a:
    multi_inputs[0].append(&input_field_a_col1);
    multi_inputs[0].append(&input_field_a_col2);

    // For input_b:
    multi_inputs[1].append(&input_field_b_col1);
    multi_inputs[1].append(&input_field_b_col2);

    // For output_c:
    multi_outputs[0].append(&output_field_c_col1);
    multi_outputs[0].append(&output_field_c_col2);


    Info << "Creating TF_OF_Predictor\n" << endl;
    TF_OF_Predictor pd = TF_OF_Predictor("constant/TF_Predictor_Dict", "model_pb");


    Info << "Running the models \n" << endl;
    pd.predict(multi_inputs, multi_outputs);

    Info << "Print the running results\n" << endl;
    Info << "Results of column 1: \n"
         << input_field_a_col1 << " + " << input_field_b_col1 << " = " << output_field_c_col1 << endl;

    Info << "Results from SavedModel format: \n"
         << input_field_a_col2 << " + " << input_field_b_col2 << " = " << output_field_c_col2 << endl;

    return 0;
}
