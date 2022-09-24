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

Class
    TF_OF_Predictor

Description
    Library to deploy neural network models using OpenFOAM "volScalarField" 
    or "scalarField" as input and output

SourceFiles
    TF_OF_Predictor.C


\*---------------------------------------------------------------------------*/


#ifndef TF_OF_PREDICTOR_H
#define TF_OF_PREDICTOR_H


#include <string>
#include <vector>

#include "scalarField.H"
#include "volFieldsFwd.H"

class TF_OF_Predictor_Impl;
class TF_OF_Predictor {
public:
    explicit TF_OF_Predictor(std::string Dict_dir, std::string Model_name);
    explicit TF_OF_Predictor(std::string Model_name);
    explicit TF_OF_Predictor();

    virtual ~TF_OF_Predictor();

    void predict(Foam::List<Foam::volScalarField*>& inputs, Foam::List<Foam::volScalarField*>& outputs);
    void predict(Foam::List<Foam::List<Foam::volScalarField * > > & multi_inputs, Foam::List<Foam::List<Foam::volScalarField * > > & multi_outputs);

    void predict(Foam::List<Foam::scalarField * > & inputs, Foam::List<Foam::scalarField * > & outputs);
    void predict(Foam::List<Foam::List<Foam::scalarField * > > & multi_inputs, Foam::List<Foam::List<Foam::scalarField * > > & multi_outputs);

   private:
    TF_OF_Predictor_Impl* d; 
};


#endif  // TF_OF_PREDICTOR_H
