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
    laplacianFoam_Ideal

Description
    Solves a simple Laplace equation with an analytical emissivity source term, 
    e.g. for thermal diffusion in a solid.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "simpleControl.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCase.H"

    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating temperature distribution\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // eps.write();

        while (simple.correctNonOrthogonal())
        {
            solve
            (
                fvm::ddt(T) - fvm::laplacian(DT, T) 
                == 
                eps * (pow(T_inf, 4) - pow(T, 4))
                +
                h * (T_inf - T)
            );
        }

        eps = eps_ref * 
        ( 1.0 
        + 5.0 * Foam::sin((3 * Foam::constant::mathematical::pi /200.0) * (T/T_ref)) 
        + Foam::exp(0.02*(T/T_ref))
        ) 
        * 0.0001;
        
        

        // #include "write.H"

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
