    Info<< "Reading field T\n" << endl;

    volScalarField T
    (
        IOobject
        (
            "T",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    


    Info<< "Reading transportProperties\n" << endl;

    IOdictionary transportProperties
    (
        IOobject
        (
            "transportProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    );


    Info<< "Reading diffusivity DT\n" << endl;

    dimensionedScalar DT
    (
        transportProperties.lookup("DT")
    );

    Info << "Reading T_inf \n" << endl;

    dimensionedScalar T_inf(
        transportProperties.lookup("T_inf")
    );


    Info << "Reading h \n" << endl;

    dimensionedScalar h(
        transportProperties.lookup("h")
    );


    Info << "Reading T_ref \n" << endl;

    dimensionedScalar T_ref(
        transportProperties.lookup("T_ref")
    );




    Info << "Reading eps_ref \n" << endl;

    dimensionedScalar eps_ref(
        transportProperties.lookup("eps_ref")
    );


    Info<< "Creating field eps \n" << endl;


    volScalarField eps
    (
        IOobject
        (
            "eps",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        eps_ref * (1.0 + 5.0 * Foam::sin((3 * Foam::constant::mathematical::pi /200) * (T/T_ref)) + Foam::exp(0.02*(T/T_ref))) * 0.0001
    );