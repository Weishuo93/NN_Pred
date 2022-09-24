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
    Foam::RASModels::kOmegaSST_nutML_incompressible

SourceFiles
    kOmegaSST_nutML_incompressible.C

\*---------------------------------------------------------------------------*/

#include "kOmegaSST_nutML_incompressible.H"

#include "addToRunTimeSelectionTable.H"

#include "bound.H"
#include "fvOptions.H"
#include "wallDist.H"
// #include "nutkWallFunctionFvPatchScalarField.H"
// #include "wallFvPatch.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace incompressible
{
namespace RASModels
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(kOmegaSST_nutML_incompressible, 0);
addToRunTimeSelectionTable(RASModel, kOmegaSST_nutML_incompressible, dictionary);

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //


tmp<volScalarField> kOmegaSST_nutML_incompressible::F1
(
    const volScalarField& CDkOmega
) const
{
    tmp<volScalarField> CDkOmegaPlus = max
    (
        CDkOmega,
        dimensionedScalar("1.0e-10", dimless/sqr(dimTime), 1.0e-10)
    );

    tmp<volScalarField> arg1 = min
    (
        min
        (
            max
            (
                (scalar(1)/betaStar_)*sqrt(k_)/(omega_*y_),
                scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
            ),
            (4*alphaOmega2_)*k_/(CDkOmegaPlus*sqr(y_))
        ),
        scalar(10)
    );

    return tanh(pow4(arg1));
}


tmp<volScalarField> kOmegaSST_nutML_incompressible::F2() const
{
    tmp<volScalarField> arg2 = min
    (
        max
        (
            (scalar(2)/betaStar_)*sqrt(k_)/(omega_*y_),
            scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
        ),
        scalar(100)
    );

    return tanh(sqr(arg2));
}


tmp<volScalarField> kOmegaSST_nutML_incompressible::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*(this->mu()/this->rho_)/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}


tmp<volScalarField> kOmegaSST_nutML_incompressible::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}



void kOmegaSST_nutML_incompressible::correctNut
(
    const volScalarField& S2,
    const volScalarField& F2
)
{
    this->nut_ = a1_*k_/max(a1_*omega_, b1_*F2*sqrt(S2));
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    // RASModel::correctNut();
    // eddyViscosity<incompressible::RASModel>::correctNut();
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //


void kOmegaSST_nutML_incompressible::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))), F23());
}



tmp<volScalarField> kOmegaSST_nutML_incompressible::epsilonByk
(
    const volScalarField& F1,
    const volScalarField& F2
) const
{
    return betaStar_*omega_;
}



tmp<fvScalarMatrix>
kOmegaSST_nutML_incompressible::kSource() const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            k_,
            dimVolume*this->rho_.dimensions()*k_.dimensions()/dimTime
        )
    );
}



tmp<fvScalarMatrix>
kOmegaSST_nutML_incompressible::omegaSource() const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            omega_,
            dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
        )
    );
}



tmp<fvScalarMatrix> kOmegaSST_nutML_incompressible::Qsas
(
    const volScalarField& S2,
    const volScalarField& gamma,
    const volScalarField& beta
) const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            omega_,
            dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
        )
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //
kOmegaSST_nutML_incompressible::kOmegaSST_nutML_incompressible
(
    const geometricOneField& alpha,
    const geometricOneField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<incompressible::RASModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),
    
    alphaK1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK1",
            this->coeffDict_,
            0.85
        )
    ),
    alphaK2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK2",
            this->coeffDict_,
            1.0
        )
    ),
    alphaOmega1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega1",
            this->coeffDict_,
            0.5
        )
    ),
    alphaOmega2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega2",
            this->coeffDict_,
            0.856
        )
    ),
    gamma1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma1",
            this->coeffDict_,
            5.0/9.0
        )
    ),
    gamma2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma2",
            this->coeffDict_,
            0.44
        )
    ),
    beta1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta1",
            this->coeffDict_,
            0.075
        )
    ),
    beta2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta2",
            this->coeffDict_,
            0.0828
        )
    ),
    betaStar_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    a1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "a1",
            this->coeffDict_,
            0.31
        )
    ),
    b1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "b1",
            this->coeffDict_,
            1.0
        )
    ),
    c1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "c1",
            this->coeffDict_,
            10.0
        )
    ),
    F3_
    (
        Switch::lookupOrAddToDict
        (
            "F3",
            this->coeffDict_,
            false
        )
    ),

    y_(wallDist::New(this->mesh_).y()),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    Ndy_ ( 
        IOobject
        (
            "Ndy",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        (sqrt(k_) * y_) / (50.0 * this->nu())
    ), 

    Ret_(
        IOobject
        (
            "Ret",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        Ndy_
    ), 

    Q1_Kkmean_ND_ ( 
        IOobject
        (
            "Q1_Kkmean_ND",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        Ndy_
    ), 

    Q2_Ret_ND_(
        IOobject
        (
            "Q2_Ret_ND",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        Ndy_
    ), 

    nut_ML_ND_(
        IOobject
        (
            "nut_ML_ND",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        Ndy_
    ),

    nut_ML_(
        IOobject
        (
            "nut_ML",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.* (this->nut_)
    ),

    nut_RANS_(
        IOobject
        (
            "nut_RANS",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.* (this->nut_)
    ),

    nut_U_(
        IOobject
        (
            "nut_U",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    nut_U_old_(
        IOobject
        (
            "nut_U_old",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.* (this->nut_)
    )


{
    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);
    if (type == typeName)
    {
        this->printCoeffs(type);
        Info << "Weighted Machine-Learning k-Omega-nut Turbulence Model." << endl;

        dimensionedScalar smallnutU_("smallnutU", dimensionSet(0, 2, -1, 0, 0, 0, 0), 2e-08);
        dimensionedScalar smallUsquare_(" smallUsquare", dimensionSet(0, 2, -2, 0, 0, 0, 0), 1e-08);
        dimensionedScalar smallOmega_("smallOmega", dimensionSet(0, 0, -1, 0, 0, 0, 0), 1e-08);

        Ndy_  =   (sqrt(k_) * y_) / (50.0 * this->nu()) ; // = sqrt(k_dns).*ySample./nu./50;

        Ret_ = (k_) / (50.0 * ((this->nu() * omega_) + smallUsquare_));

        Q1_Kkmean_ND_ = (-5.0 * pow(min(15.0 * Ndy_, 1.0), 6 ) + 6.0 * pow( min( 15.0 *  Ndy_, 1.0 ),5) ) * 25.0* (k_) / ( 0.5*((this->U_) & (this->U_))  + 25.0* k_ + smallUsquare_) ;



        Q2_Ret_ND_ = (k_) / (50.0 * (this->nu() * omega_) + (k_) + smallUsquare_ ) ;


        inputs_.append(&Q1_Kkmean_ND_);
        inputs_.append(&Q2_Ret_ND_);

        outputs_.append(&nut_ML_ND_);

        predictor.predict(inputs_, outputs_);




        nut_ML_ND_ = min(max(nut_ML_ND_, 0.0),0.95);
        nut_ML_ = 30.0 * ( this->nu() ) * ((nut_ML_ND_ * Ret_)/(1. - nut_ML_ND_)) ; 


        nut_RANS_ = this->nut_;


        nut_U_ = nut_ML_ ;

        nut_U_old_ = nut_ML_ ;

        nut_U_.correctBoundaryConditions();


    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool kOmegaSST_nutML_incompressible::read()
{
    if (eddyViscosity<incompressible::RASModel>::read())
    {
        alphaK1_.readIfPresent(this->coeffDict());
        alphaK2_.readIfPresent(this->coeffDict());
        alphaOmega1_.readIfPresent(this->coeffDict());
        alphaOmega2_.readIfPresent(this->coeffDict());
        gamma1_.readIfPresent(this->coeffDict());
        gamma2_.readIfPresent(this->coeffDict());
        beta1_.readIfPresent(this->coeffDict());
        beta2_.readIfPresent(this->coeffDict());
        betaStar_.readIfPresent(this->coeffDict());
        a1_.readIfPresent(this->coeffDict());
        b1_.readIfPresent(this->coeffDict());
        c1_.readIfPresent(this->coeffDict());
        F3_.readIfPresent("F3", this->coeffDict());

        return true;
    }
    else
    {
        return false;
    }
}


tmp<fvVectorMatrix> kOmegaSST_nutML_incompressible::divDevRhoReff(volVectorField& U) const
{
    Info << "Using divDevRhoReff from kOmegaSST_nutML_incompressible" << endl;
    return
    (
        // this->alpha_*this->rho_*this->gradTau_
 	  - fvc::div((this->alpha_*this->rho_*(this->nuUEff()))*dev2(T(fvc::grad(U))))
 	  - fvm::laplacian(this->alpha_*this->rho_*(this->nuUEff()), U)
    );
}


void kOmegaSST_nutML_incompressible::correct()
{
    dimensionedScalar smallnutU_("smallnutU", dimensionSet(0, 2, -1, 0, 0, 0, 0), 2e-08);
    dimensionedScalar smallUsquare_(" smallUsquare", dimensionSet(0, 2, -2, 0, 0, 0, 0), 1e-08);
    dimensionedScalar smallOmega_("smallOmega", dimensionSet(0, 0, -1, 0, 0, 0, 0), 1e-08);


    if (!this->turbulence_)
    {
        this->nut_ = k_/(omega_+smallOmega_);
        this->nut_.correctBoundaryConditions();
        // eddyViscosity<incompressible::RASModel>::correctNut();
        return;
    }

     // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    volScalarField& nut = this->nut_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    // BasicTurbulenceModel::correct();
    eddyViscosity<incompressible::RASModel>::correct();

    volScalarField divU(fvc::div(fvc::absolute(this->phi(), U)));

    tmp<volTensorField> tgradU = fvc::grad(U);
    volScalarField S2(2*magSqr(symm(tgradU())));
    volScalarField GbyNu((tgradU() && dev(twoSymm(tgradU()))));
    volScalarField G(this->GName(), nut*GbyNu);
    tgradU.clear();

    // Update omega and G at the wall
    omega_.boundaryFieldRef().updateCoeffs();

    volScalarField CDkOmega
    (
        (2*alphaOmega2_)*(fvc::grad(k_) & fvc::grad(omega_))/omega_
    );

    volScalarField F1(this->F1(CDkOmega));
    volScalarField F23(this->F23());

    {
        volScalarField gamma(this->gamma(F1));
        volScalarField beta(this->beta(F1));

        // Turbulent frequency equation
        tmp<fvScalarMatrix> omegaEqn
        (
            fvm::ddt(alpha, rho, omega_)
          + fvm::div(alphaRhoPhi, omega_)
          - fvm::laplacian(alpha*rho*DomegaEff(F1), omega_)
         ==
            alpha*rho*gamma
           *min
            (
                GbyNu,
                (c1_/a1_)*betaStar_*omega_*max(a1_*omega_, b1_*F23*sqrt(S2))
            )
          - fvm::SuSp((2.0/3.0)*alpha*rho*gamma*divU, omega_)
          - fvm::Sp(alpha*rho*beta*omega_, omega_)
          - fvm::SuSp
            (
                alpha*rho*(F1 - scalar(1))*CDkOmega/omega_,
                omega_
            )
          + Qsas(S2, gamma, beta)
          + omegaSource()
          + fvOptions(alpha, rho, omega_)
        );

        omegaEqn.ref().relax();
        fvOptions.constrain(omegaEqn.ref());
        omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
        solve(omegaEqn);
        fvOptions.correct(omega_);
        bound(omega_, this->omegaMin_);
    }

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(alpha, rho, k_)
      + fvm::div(alphaRhoPhi, k_)
      - fvm::laplacian(alpha*rho*DkEff(F1), k_)
     ==
        min(alpha*rho*G, (c1_*betaStar_)*alpha*rho*k_*omega_)
      - fvm::SuSp((2.0/3.0)*alpha*rho*divU, k_)
      - fvm::Sp(alpha*rho*epsilonByk(F1, F23), k_)
      + kSource()
      + fvOptions(alpha, rho, k_)
    );

    kEqn.ref().relax();
    fvOptions.constrain(kEqn.ref());
    solve(kEqn);
    fvOptions.correct(k_);
    bound(k_, this->kMin_);

    correctNut(S2, F23);




    Ndy_  =   (sqrt(k_) * y_) / (50.0 * this->nu()) ; // = sqrt(k_dns).*ySample./nu./50;
    Ret_ = (k_) / (50.0 * ((this->nu() * omega_) + smallUsquare_));

    Q1_Kkmean_ND_ = (-5.0 * pow(min(15.0 * Ndy_, 1.0), 6 ) + 6.0 * pow( min( 15.0 *  Ndy_, 1.0 ),5) ) * 25.0* (k_) / ( 0.5*((this->U_) & (this->U_))  + 25.0* k_ + smallUsquare_) ;

    Q2_Ret_ND_ = (k_) / (50.0 * (this->nu() * omega_) + (k_) + smallUsquare_ ) ;



    predictor.predict(inputs_, outputs_);


    nut_ML_ND_ = min(max(nut_ML_ND_, 0.0), 0.95);
    nut_ML_ = 30.0 * (this->nu()) * ((nut_ML_ND_ * Ret_) / (1. - nut_ML_ND_));

    nut_RANS_ = this->nut_;

    nut_U_ = nut_ML_;
    


    Info << "nut_U Residual is " << (max(mag(nut_U_old_-nut_U_))/max(max(mag(nut_U_old_),smallnutU_))).value() << endl;

    nut_U_old_ = nut_U_;

    nut_U_.correctBoundaryConditions();

    
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace incompressible
} // End namespace Foam

// ************************************************************************* //
