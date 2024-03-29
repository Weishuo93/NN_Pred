/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2016 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::RASModels::nut_Dev

Group
    grpRASTurbulence

See also
    Foam::kOmegaSST

SourceFiles
    nut_Dev.C

\*---------------------------------------------------------------------------*/

#ifndef nut_Dev_H
#define nut_Dev_H

//TensorFlow Interface HEADERS
#include "TF_OF_Predictor.H"

//OpenFOAM HEADERS
// #include "kOmegaSSTBase.H"
// #include "RASModel.H"
#include "turbulentTransportModel.H"
#include "eddyViscosity.H"
// smoothfields
// #include "simpleFilter.H"



// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

namespace incompressible
{

namespace RASModels
{

/*---------------------------------------------------------------------------*\
                          Class nut_Dev Declaration
\*---------------------------------------------------------------------------*/


class nut_Dev
:
 public eddyViscosity<incompressible::RASModel>
{

protected:

// Original kw SST model:
    // Protected data

        // Model coefficients

            dimensionedScalar alphaK1_;
            dimensionedScalar alphaK2_;

            dimensionedScalar alphaOmega1_;
            dimensionedScalar alphaOmega2_;

            dimensionedScalar gamma1_;
            dimensionedScalar gamma2_;

            dimensionedScalar beta1_;
            dimensionedScalar beta2_;

            dimensionedScalar betaStar_;

            dimensionedScalar a1_;
            dimensionedScalar b1_;
            dimensionedScalar c1_;

            Switch F3_;



        // Fields

            //- Wall distance
            //  Note: different to wall distance in parent RASModel
            //  which is for near-wall cells only
            const volScalarField& y_;

            volScalarField k_;
            volScalarField omega_;

        // Private Member Functions

            tmp<volScalarField> F1(const volScalarField& CDkOmega) const;
            tmp<volScalarField> F2() const;
            tmp<volScalarField> F3() const;
            tmp<volScalarField> F23() const;

            tmp<volScalarField> blend
            (
                const volScalarField& F1,
                const dimensionedScalar& psi1,
                const dimensionedScalar& psi2
            ) const
            {
                return F1*(psi1 - psi2) + psi2;
            }

            tmp<volScalarField> alphaK(const volScalarField& F1) const
            {
                return blend(F1, alphaK1_, alphaK2_);
            }

            tmp<volScalarField> alphaOmega(const volScalarField& F1) const
            {
                return blend(F1, alphaOmega1_, alphaOmega2_);
            }

            tmp<volScalarField> beta(const volScalarField& F1) const
            {
                return blend(F1, beta1_, beta2_);
            }

            tmp<volScalarField> gamma(const volScalarField& F1) const
            {
                return blend(F1, gamma1_, gamma2_);
            }

            void correctNut(const volScalarField& S2, const volScalarField& F2);


        // Protected Member Functions

            virtual void correctNut();

            //- Return epsilon/k which for standard RAS is betaStar*omega
            virtual tmp<volScalarField> epsilonByk
            (
                const volScalarField& F1,
                const volScalarField& F2
            ) const;

            virtual tmp<fvScalarMatrix> kSource() const;

            virtual tmp<fvScalarMatrix> omegaSource() const;

            virtual tmp<fvScalarMatrix> Qsas
            (
                const volScalarField& S2,
                const volScalarField& gamma,
                const volScalarField& beta
            ) const;



    // Fields for ML 

            volScalarField tau_;
            volTensorField gradU_;
            volSymmTensorField S_;
            volTensorField W_;

            volVectorField gradk_;
            volVectorField gradomega_;

            volScalarField Lk_;

            volScalarField Pk_;
            volScalarField Dk_;
            volScalarField Tk_;

            volScalarField Pomg_;
            volScalarField Domg_;
            volScalarField Comg_;
            volScalarField Tomg_;

            volScalarField Sd2byNu_;
            volScalarField Ssqr_;
            volScalarField SqrK_;
            volScalarField GsqrK_;

            

            volScalarField Ndy_; //= k_dns./(0.5*sum(U_dns.^2,2)+k_dns);
            volScalarField Ret_;

            volScalarField Kafangk_;
            volScalarField fBeta_;


            

            volScalarField R_Weight_ini_;
            volScalarField R_Weight_;






            volScalarField Q1_Kkmean_ND_; // = sqrt(k_dns).*ySample./nu./50;
            volScalarField Q2_Ret_ND_; //=k_dns/omega/nu;
            volScalarField Q3_Ndy_ND_;
            volScalarField Q4_PDK_ND_;
            volScalarField Q5_PDO_ND_;
            volScalarField Q6_fBeta_ND_;
            volScalarField Q7_Sd2byNu_ND_;
            volScalarField Q8_GsqrK_ND_;


            // volScalarField R_Weight_ini_;
            // volScalarField R_Weight_;

            volScalarField nut_ML_ND_;
            volScalarField nut_ML_;
            volScalarField nut_RANS_; // = sqrt(k_dns.*omega_dns.^2);
            volScalarField nut_U_;
            volScalarField nut_U_old_;

    // TF_Predictor for ML

            List<volScalarField*> inputs_;
            List<volScalarField*> outputs_;
            TF_OF_Predictor predictor;



public:




    //- Runtime type information
    TypeName("nut_Dev");


    // Constructors

        //- Construct from components
        nut_Dev
        (
            const geometricOneField& alpha,
            const geometricOneField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~nut_Dev()
    {}

    //Member Functions

        //- Read RASProperties dictionary
        virtual bool read();

        //- Return the effective diffusivity for k
        tmp<volScalarField> DkEff(const volScalarField& F1) const
        {
            return tmp<volScalarField>
            (
                new volScalarField("DkEff", alphaK(F1)*this->nut_ + this->nu())
            );
        }

        //- Return the effective diffusivity for omega
        tmp<volScalarField> DomegaEff(const volScalarField& F1) const
        {
            return tmp<volScalarField>
            (
                new volScalarField
                (
                    "DomegaEff",
                    alphaOmega(F1)*this->nut_ + this->nu()
                )
            );
        }

        //- Return the turbulence kinetic energy
        virtual tmp<volScalarField> k() const
        {
            return k_;
        }

        //- Return the turbulence kinetic energy dissipation rate
        virtual tmp<volScalarField> epsilon() const
        {
                return tmp<volScalarField>
            (
                new volScalarField
                (
                    IOobject
                    (
                        "epsilon",
                        this->mesh_.time().timeName(),
                        this->mesh_
                    ),
                    betaStar_*k_*omega_,
                    omega_.boundaryField().types()
                )
            );
        }

        //- Return the turbulence kinetic energy dissipation rate
        virtual tmp<volScalarField> omega() const
        {
            return omega_;
        }

        //- Return the source term for the momentum equation
        virtual tmp<fvVectorMatrix> divDevRhoReff(volVectorField& U) const;

        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct();

        tmp<volScalarField> nuUEff() const
        {
            return tmp<volScalarField>
            (
                new volScalarField
                (
                    IOobject::groupName("nuUEff", this->U_.group()),
                    this->nut_U() + this->nu()
                )
            );
        }


        tmp<volScalarField> nut_U() const
        {
            return nut_U_;
        }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace incompressible
} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#endif

// ************************************************************************* //
