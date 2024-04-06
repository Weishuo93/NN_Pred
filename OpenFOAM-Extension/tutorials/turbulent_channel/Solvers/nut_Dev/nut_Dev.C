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

\*---------------------------------------------------------------------------*/

#include "nut_Dev.H"

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

defineTypeNameAndDebug(nut_Dev, 0);
addToRunTimeSelectionTable(RASModel, nut_Dev, dictionary);

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //


tmp<volScalarField> nut_Dev::F1
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


tmp<volScalarField> nut_Dev::F2() const
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


tmp<volScalarField> nut_Dev::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*(this->mu()/this->rho_)/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}


tmp<volScalarField> nut_Dev::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}



void nut_Dev::correctNut
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


void nut_Dev::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))), F23());
}



tmp<volScalarField> nut_Dev::epsilonByk
(
    const volScalarField& F1,
    const volScalarField& F2
) const
{
    return betaStar_*omega_;
}



tmp<fvScalarMatrix>
nut_Dev::kSource() const
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
nut_Dev::omegaSource() const
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



tmp<fvScalarMatrix> nut_Dev::Qsas
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
// const dimensionedScalar smallOmega_ ("smallOmega",dimensionSet (0,0,-1,0,0,0,0), 1e-9);
// const dimensionedScalar smallDimless_ ("smallDimless",dimless, 1e-9);
// const dimensionedScalar smallDistance_ ("smallDistance_",dimensionSet (0,1,0,0,0,0,0), 1e-8);
// const dimensionedScalar smallnutU_ ("smallnutU",dimensionSet (0,2,-1,0,0,0,0), 2e-05);
nut_Dev::nut_Dev
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

    tau_ ( 1./ (this->omega_) ),
    gradU_( fvc::grad(this->U_)-(1./3.)*I*tr(fvc::grad(this->U_)) ),
    S_( tau_*symm(gradU_.T())),
    W_( tau_*skew(gradU_.T())),

    gradk_(fvc::grad(this->k_)),
    gradomega_(fvc::grad(this->omega_)),

    Lk_(
        IOobject
        (
            "Lk",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        fvc::laplacian(this->k_)
    ),

    Pk_(
        IOobject
        (
            "Pk",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        (this->alpha_)*(this->rho_)*(this->betaStar_) * (this->omega_) *(this->k_)
    ),

    Dk_(
        IOobject
        (
            "Dk",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        (this->alpha_)*(this->rho_)*(this->betaStar_) * (this->omega_) *(this->k_)
    ),
    

    Tk_(
        IOobject
        (
            "Tk",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        (this->alpha_)*(this->rho_)*(this->betaStar_) * (this->omega_) *(this->k_)
    ),

    Pomg_(
        IOobject
        (
            "Pomg",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        (this->alpha_)*(this->rho_)*(this->omega_)*(this->omega_)
    ),

    Domg_(
        IOobject
        (
            "Domg",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        (this->alpha_)*(this->rho_)*(this->omega_)*(this->omega_)
    ),

    Comg_(
        IOobject
        (
            "Comg",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        (this->alpha_)*(this->rho_)*(this->omega_)*(this->omega_)
    ),

    Tomg_(
        IOobject
        (
            "Tomg",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        (this->alpha_)*(this->rho_)*(this->omega_)*(this->omega_)
    ),

    Sd2byNu_(
        IOobject
        (
            "Sd2byNu",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.0 * tr(S_&S_)
    ),

    Ssqr_(
        IOobject
        (
            "Ssqr",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        ((symm(gradU_.T())) && (symm(gradU_.T())))
    ),

    SqrK_(
        IOobject
        (
            "SqrK",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        sqrt(this->k_)
    ),

    GsqrK_(
        IOobject
        (
            "GsqrK",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        sqrt(fvc::grad(SqrK_) & fvc::grad(SqrK_)) / (this->omega_)
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

    Kafangk_(
        IOobject
        (
            "Kafangk",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        Ndy_
    ),



    fBeta_(
        IOobject
        (
            "fBeta",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        Ndy_
    ),

    R_Weight_ini_( 0.0 * Ndy_ ),

    R_Weight_(
        IOobject
        (
            "R_Weight_",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        1. - R_Weight_ini_*R_Weight_ini_*R_Weight_ini_*R_Weight_ini_*R_Weight_ini_
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

    Q3_Ndy_ND_(
        IOobject
        (
            "Q3_Ndy_ND",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        Ndy_
    ), 

    Q4_PDK_ND_(
        IOobject
        (
            "Q4_PDK_ND",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        Ndy_
    ), 

    Q5_PDO_ND_(
        IOobject
        (
            "Q5_PDO_ND",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        Ndy_
    ), 

    Q6_fBeta_ND_(
        IOobject
        (
            "Q6_fBeta_ND",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        Ndy_
    ), 

    Q7_Sd2byNu_ND_(
        IOobject
        (
            "Q7_Sd2byNu_ND",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        Ndy_
    ), 

    Q8_GsqrK_ND_(
        IOobject
        (
            "Q8_GsqrK_ND",
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
            IOobject::NO_WRITE
        ),
        0.* (this->nut_)
    )

{
    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);
    if (type == typeName)
    {
        this->printCoeffs(type);
        Info << "Weighted Machine-Learning nut_Dev Turbulence Model." << endl;

        dimensionedScalar smallnutU_("smallnutU", dimensionSet(0, 2, -1, 0, 0, 0, 0), 2e-08);
        dimensionedScalar smallUsquare_(" smallUsquare", dimensionSet(0, 2, -2, 0, 0, 0, 0), 1e-08);
        dimensionedScalar smallOmega_("smallOmega", dimensionSet(0, 0, -1, 0, 0, 0, 0), 1e-08);
        dimensionedScalar smallPk_("smallPk", dimensionSet(0, 2, -3, 0, 0, 0, 0), 1e-08);
        dimensionedScalar smallPomg_("smallPomg", dimensionSet(0, 0, -2, 0, 0, 0, 0), 1e-08);


        IOdictionary nut_Dev_Dict
        (
            IOobject
            (
                "nut_Dev_Dict",
                (this->mesh_).time().constant(),
                this->mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            )
        );
        // Print Information of Current choice of Features

#if defined(OPENFOAM) && OPENFOAM >= 1906
        Switch UseQ1("UseQ1", nut_Dev_Dict);
        Info << "UseQ1 is " << UseQ1 << endl;
        Switch UseQ2("UseQ2", nut_Dev_Dict);
        Info << "UseQ2 is " << UseQ2 << endl;
        Switch UseQ3("UseQ3", nut_Dev_Dict);
        Info << "UseQ3 is " << UseQ3 << endl;
        Switch UseQ4("UseQ4", nut_Dev_Dict);
        Info << "UseQ4 is " << UseQ4 << endl;
        Switch UseQ5("UseQ5", nut_Dev_Dict);
        Info << "UseQ5 is " << UseQ5 << endl;
        Switch UseQ6("UseQ6", nut_Dev_Dict);
        Info << "UseQ6 is " << UseQ6 << endl;
        Switch UseQ7("UseQ7", nut_Dev_Dict);
        Info << "UseQ7 is " << UseQ7 << endl;
        Switch UseQ8("UseQ8", nut_Dev_Dict);
        Info << "UseQ8 is " << UseQ8 << endl;
#else
        Switch UseQ1(nut_Dev_Dict.lookup("UseQ1"));
        Info << "UseQ1 is " << UseQ1 << endl;
        Switch UseQ2(nut_Dev_Dict.lookup("UseQ2"));
        Info << "UseQ2 is " << UseQ2 << endl;
        Switch UseQ3(nut_Dev_Dict.lookup("UseQ3"));
        Info << "UseQ3 is " << UseQ3 << endl;
        Switch UseQ4(nut_Dev_Dict.lookup("UseQ4"));
        Info << "UseQ4 is " << UseQ4 << endl;
        Switch UseQ5(nut_Dev_Dict.lookup("UseQ5"));
        Info << "UseQ5 is " << UseQ5 << endl;
        Switch UseQ6(nut_Dev_Dict.lookup("UseQ6"));
        Info << "UseQ6 is " << UseQ6 << endl;
        Switch UseQ7(nut_Dev_Dict.lookup("UseQ7"));
        Info << "UseQ7 is " << UseQ7 << endl;
        Switch UseQ8(nut_Dev_Dict.lookup("UseQ8"));
        Info << "UseQ8 is " << UseQ8 << endl;
#endif




        tau_ = 1./ ( this->omega_ ) ;
        gradU_ = fvc::grad(this->U_)-(1./3.)*I*tr(fvc::grad(this->U_)) ;
        S_ = tau_*symm(gradU_.T());
        W_ = tau_*skew(gradU_.T());

        gradk_ = fvc::grad(this->k_);
        gradomega_ = fvc::grad(this->omega_);




        Lk_ = fvc::laplacian(this->k_);
        Ssqr_ = ((symm(gradU_.T())) && (symm(gradU_.T())));
        Sd2byNu_ = (this->y_) * (this->y_) * sqrt(Ssqr_) / (this->nu());


        SqrK_ = sqrt(this->k_);
        GsqrK_ = sqrt(fvc::grad(SqrK_) & fvc::grad(SqrK_)) / (this->omega_ + smallOmega_);

        // Kafangk_ = (gradk_ & gradomega_) / (this->omega_*this->omega_*this->omega_ );
        Kafangk_ = max((gradk_ & gradomega_) / (this->omega_*this->omega_*this->omega_ ), 0.0);
        fBeta_ = max(((1.0 + 680 * Kafangk_ * Kafangk_) / (1.0 + 400 * Kafangk_ * Kafangk_) - 1.0), 0.0);




        Ndy_  =   (sqrt(k_) * y_) / (50.0 * this->nu()) ; // = sqrt(k_dns).*ySample./nu./50;
        Ret_ = (k_) / (50.0 * ((this->nu() * omega_) + smallUsquare_));

        Q1_Kkmean_ND_ = (-5.0 * pow(min(15.0 * Ndy_, 1.0), 6 ) + 6.0 * pow( min( 15.0 *  Ndy_, 1.0 ),5) ) * 25.0* (k_) / ( 0.5*((this->U_) & (this->U_))  + 25.0* k_ + smallUsquare_) ;
        Q2_Ret_ND_ = (k_) / (50.0 * (this->nu() * omega_) + (k_) + smallUsquare_ ) ;
        Q3_Ndy_ND_ = (0.5 * Ndy_)/(1.0 + 0.5 * Ndy_);
        Q4_PDK_ND_ = (Pk_)/(mag(Pk_) + mag(Dk_) + smallPk_ );
        Q5_PDO_ND_ = (Pomg_)/(mag(Pomg_) + mag(Domg_) + smallPomg_ );
        Q6_fBeta_ND_ = 5.0 * fBeta_ / (5.0 * fBeta_ + 1.0);
        Q7_Sd2byNu_ND_ = Sd2byNu_ / (Sd2byNu_ + 5000.0);
        Q8_GsqrK_ND_ =  GsqrK_ / (GsqrK_ + 1.0);



        // predictor = TF_OF_Predictor();



        if (UseQ1) {
            inputs_.append(&Q1_Kkmean_ND_);
        }
        if (UseQ2) {
            inputs_.append(&Q2_Ret_ND_);
        }
        if (UseQ3) {
            inputs_.append(&Q3_Ndy_ND_);
        }
        if (UseQ4) {
            inputs_.append(&Q4_PDK_ND_);
        }
        if (UseQ5) {
            inputs_.append(&Q5_PDO_ND_);
        }
        if (UseQ6) {
            inputs_.append(&Q6_fBeta_ND_);
        }
        if (UseQ7) {
            inputs_.append(&Q7_Sd2byNu_ND_);
        }
        if (UseQ8) {
            inputs_.append(&Q8_GsqrK_ND_);
        }

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

bool nut_Dev::read()
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


tmp<fvVectorMatrix> nut_Dev::divDevRhoReff(volVectorField& U) const
{
    //Info << "Using divDevRhoReff from BSL-EARSM" << endl;
    return
    (
        // this->alpha_*this->rho_*this->gradTau_
 	  - fvc::div((this->alpha_*this->rho_*(this->nuUEff()))*dev2(T(fvc::grad(U))))
 	  - fvm::laplacian(this->alpha_*this->rho_*(this->nuUEff()), U)
    );
}


void nut_Dev::correct()
{
    dimensionedScalar smallnutU_("smallnutU", dimensionSet(0, 2, -1, 0, 0, 0, 0), 2e-08);
    dimensionedScalar smallUsquare_(" smallUsquare", dimensionSet(0, 2, -2, 0, 0, 0, 0), 1e-08);
    dimensionedScalar smallOmega_("smallOmega", dimensionSet(0, 0, -1, 0, 0, 0, 0), 1e-08);
    dimensionedScalar smallPk_("smallPk", dimensionSet(0, 2, -3, 0, 0, 0, 0), 1e-08);
    dimensionedScalar smallPomg_("smallPomg", dimensionSet(0, 0, -2, 0, 0, 0, 0), 1e-08);


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

    // Up is traditional k-Omega-SST Model



    tau_ = 1./ ( this->omega_ ) ;
    gradU_ = fvc::grad(this->U_)-(1./3.)*I*tr(fvc::grad(this->U_)) ;
    S_ = tau_*symm(gradU_.T());
    W_ = tau_*skew(gradU_.T());

    gradk_ = fvc::grad(this->k_);
    gradomega_ = fvc::grad(this->omega_);




    Lk_ = fvc::laplacian(this->k_);
    Ssqr_ = ((symm(gradU_.T())) && (symm(gradU_.T())));
    Sd2byNu_ = (this->y_) * (this->y_) * sqrt(Ssqr_) / (this->nu());


    SqrK_ = sqrt(this->k_);
    GsqrK_ = sqrt(fvc::grad(SqrK_) & fvc::grad(SqrK_)) / (this->omega_ + smallOmega_);

    // Kafangk_ = (gradk_ & gradomega_) / (this->omega_*this->omega_*this->omega_ );
    Kafangk_ = max((gradk_ & gradomega_) / (this->omega_*this->omega_*this->omega_ ), 0.0);
    fBeta_ = max(((1.0 + 680 * Kafangk_ * Kafangk_) / (1.0 + 400 * Kafangk_ * Kafangk_) - 1.0), 0.0);




    Ndy_  =   (sqrt(k_) * y_) / (50.0 * this->nu()) ; // = sqrt(k_dns).*ySample./nu./50;
    Ret_ = (k_) / (50.0 * ((this->nu() * omega_) + smallUsquare_));

    Q1_Kkmean_ND_ = (-5.0 * pow(min(15.0 * Ndy_, 1.0), 6 ) + 6.0 * pow( min( 15.0 *  Ndy_, 1.0 ),5) ) * 25.0* (k_) / ( 0.5*((this->U_) & (this->U_))  + 25.0* k_ + smallUsquare_) ;
    Q2_Ret_ND_ = (k_) / (50.0 * (this->nu() * omega_) + (k_) + smallUsquare_ ) ;
    Q3_Ndy_ND_ = (0.5 * Ndy_)/(1.0 + 0.5 * Ndy_);
    Q4_PDK_ND_ = (Pk_)/(mag(Pk_) + mag(Dk_) + smallPk_ );
    Q5_PDO_ND_ = (Pomg_)/(mag(Pomg_) + mag(Domg_) + smallPomg_ );
    Q6_fBeta_ND_ = 5.0 * fBeta_ / (5.0 * fBeta_ + 1.0);
    Q7_Sd2byNu_ND_ = Sd2byNu_ / (Sd2byNu_ + 5000.0);
    Q8_GsqrK_ND_ =  GsqrK_ / (GsqrK_ + 1.0);





    predictor.predict(inputs_, outputs_);


    nut_ML_ND_ = min(max(nut_ML_ND_, 0.0), 0.95);
    nut_ML_ = 30.0 * (this->nu()) * ((nut_ML_ND_ * Ret_) / (1. - nut_ML_ND_));

    nut_RANS_ = this->nut_;

    // R_Weight_ini_ =  sin(min( 3.0 * max(Ndy_ - 0.1, scalar(0.0)),Foam::constant::mathematical::piByTwo ));
    // R_Weight_ = 1. - (126.0*pow(R_Weight_ini_,10) -560.0*pow(R_Weight_ini_,9) + 945.0*pow(R_Weight_ini_,8) - 720.0*pow(R_Weight_ini_,7) + 210.0*pow(R_Weight_ini_,6)) ;


    nut_U_ = nut_ML_;

    // nut_U_ = (1. - R_Weight_) * nut_ML_ + R_Weight_ * nut_RANS_ ;

    Info << "nut_U Residual is " << (max(mag(nut_U_old_-nut_U_))/max(max(mag(nut_U_old_),smallnutU_))).value() << endl;

    nut_U_old_ = nut_U_;

    nut_U_.correctBoundaryConditions();

    
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace incompressible
} // End namespace Foam

// ************************************************************************* //
