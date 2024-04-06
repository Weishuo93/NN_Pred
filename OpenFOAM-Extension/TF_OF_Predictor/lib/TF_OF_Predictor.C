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




#include "TF_OF_Predictor.H"

#include "predictor.h"


#include "IFstream.H"
#include "dictionary.H"

#include "GeometricFields.H"
#include "volMesh.H"


#include "Switch.H"

class TF_OF_Predictor_Impl {
   public:
    explicit TF_OF_Predictor_Impl(std::string Dict_dir, std::string Model_name);
    explicit TF_OF_Predictor_Impl(std::string Model_name);
    explicit TF_OF_Predictor_Impl();
    virtual ~TF_OF_Predictor_Impl() {
        if (pd != nullptr) {
            delete pd;
        }
    }

#ifdef WM_DP
    std::vector<std::vector<double>> input_buffers;
    std::vector<std::vector<double>> output_buffers;
#elif WM_SP
    std::vector<std::vector<float>> input_buffers;
    std::vector<std::vector<float>> output_buffers;
#else
    std::vector<std::vector<double>> input_buffers;
    std::vector<std::vector<double>> output_buffers;
#endif



    Predictor* pd;

    Predictor::DataLayout layout;
    Predictor::CopyMethod method;

    Foam::fileNameList inputs_names;
    Foam::fileNameList outputs_names;

};

TF_OF_Predictor_Impl::TF_OF_Predictor_Impl(std::string Dict_dir, std::string Model_name)
    : pd(nullptr) {
    Foam::IFstream ifs(Dict_dir);
    Foam::dictionary model_dict(ifs);
    Foam::dictionary model_info;
    model_info = model_dict.subDict(Model_name);


    Foam::Switch is_pb = static_cast<Foam::Switch>(model_info.lookup("readFromPB"));

    if (is_pb) {
        Foam::fileName PB_file = static_cast<Foam::fileName>(model_info.lookup("modelDirectory"));
        this->pd = new Predictor(PB_file);
    } else {
        Foam::fileName Saved_Model_Dir =  static_cast<Foam::fileName>(model_info.lookup("modelDirectory"));
        Foam::fileName tags =  static_cast<Foam::fileName>(model_info.lookup("tags"));
        this->pd = new Predictor(Saved_Model_Dir, tags);
    }

    this->inputs_names = Foam::fileNameList(model_info.lookup("inputs"));
    this->outputs_names = Foam::fileNameList(model_info.lookup("outputs"));

    for (int i = 0; i < this->inputs_names.size(); i++) {
        Foam::Info << "Registering input node: " << this->inputs_names[i] << Foam::endl;
        this->pd->regist_node(this->inputs_names[i], Predictor::INPUT_NODE);
    }

    for (int i = 0; i < this->outputs_names.size(); i++) {
        Foam::Info << "Registering input node: " << this->outputs_names[i] << Foam::endl;
        this->pd->regist_node(this->outputs_names[i], Predictor::OUTPUT_NODE);
    }

    Foam::word layout_string =  static_cast<Foam::word>(model_info.lookup("layout"));
    if (layout_string == "ColMajor") {
        this->layout = Predictor::ColumnMajor;
    } else if (layout_string == "RowMajor") {
        this->layout = Predictor::RowMajor;
    } else {
        this->layout = Predictor::ColumnMajor;
        Foam::Info << "Failed to recognize data layout, use default ColMajor" << Foam::endl;
    }

    Foam::word method_string = static_cast<Foam::word>(model_info.lookup("copyMethod"));
    if (method_string == "Eigen") {
        this->method = Predictor::Eigen;
    } else if (method_string == "Safe") {
        this->method = Predictor::Simple;
    } else {
        this->method = Predictor::Eigen;
        Foam::Info << "Failed to recognize copy method, use default Eigen" << Foam::endl;
    }
}

TF_OF_Predictor_Impl::TF_OF_Predictor_Impl(std::string Model_name) : TF_OF_Predictor_Impl("constant/TF_Predictor_Dict", Model_name) {}

TF_OF_Predictor_Impl::TF_OF_Predictor_Impl() : TF_OF_Predictor_Impl("constant/TF_Predictor_Dict", "model" ) {}

//------------------------------------------------------------------------------------------------------------------

TF_OF_Predictor::TF_OF_Predictor(std::string Dict_dir, std::string Model_name)
    : d(nullptr) {
    d = new TF_OF_Predictor_Impl(Dict_dir, Model_name);
}

TF_OF_Predictor::TF_OF_Predictor(std::string Model_name)
    : d(nullptr) {
    d = new TF_OF_Predictor_Impl(Model_name);
}

TF_OF_Predictor::TF_OF_Predictor()
    : d(nullptr) {
    d = new TF_OF_Predictor_Impl();
}

TF_OF_Predictor::~TF_OF_Predictor() {
    if (d != nullptr) {
        delete d;
    }
}

void TF_OF_Predictor::predict(Foam::List<Foam::volScalarField*>& inputs, Foam::List<Foam::volScalarField*>& outputs) {

    if (d->inputs_names.size() == 0) {
        Foam::Info << "Input nodes not regestered " << Foam::endl;
        return;
    }
    if (d->outputs_names.size() == 0) {
        Foam::Info << "Output nodes not regestered " << Foam::endl;
        return;
    }

    if (d->inputs_names.size() != 1) {
        Foam::Info << "Not single input, please use predict(List<List<scalarField>>, List<List<scalarField>>) for multi in-out model." << Foam::endl;
        return;
    }
    if (d->outputs_names.size() != 1) {
        Foam::Info << "Not single output, please use predict(List<List<scalarField>>, List<List<scalarField>>) for multi in-out model." << Foam::endl;
        return;
    }

    if (inputs.size() == 0) {
        Foam::Info << "Empty input container" << Foam::endl;
        return;
    }

    if (outputs.size() == 0) {
        Foam::Info << "Empty output container" << Foam::endl;
        return;
    }

    int ndata = inputs[0]->size();
    if (d->pd->get_data_count() != ndata) {
        d->pd->set_data_count(ndata);
    }

    if (inputs.size() == 1) {

#ifdef WM_DP
        double* p_field = const_cast<double*>(inputs[0]->internalField().cdata());
#elif WM_SP
        float* p_field = const_cast<float*>(inputs[0]->internalField().cdata());
#else
        double* p_field = const_cast<double*>(inputs[0]->internalField().cdata());
#endif

        if (d->pd->set_node_data(d->inputs_names[0], p_field, ndata, d->layout, d->method)) {
            Foam::Info << "Succeeded set data" << Foam::endl;
        } else {
            Foam::Info << "Failed set data" << Foam::endl;
        }
    } else {
        if (d->input_buffers.empty()) {
#ifdef WM_DP
            d->input_buffers.push_back(std::vector<double>(ndata * inputs.size()));
#elif WM_SP
            d->input_buffers.push_back(std::vector<float>(ndata * inputs.size()));
#else
            d->input_buffers.push_back(std::vector<double>(ndata * inputs.size()));
#endif
            
        } else if (d->input_buffers[0].size() != static_cast<unsigned int>(ndata * inputs.size())) {
            d->input_buffers[0].resize(ndata * inputs.size());
        }

        for (int i = 0; i < inputs.size(); i++) {
#ifdef WM_DP
            double* p_field = const_cast<double*>(inputs[i]->internalField().cdata());
#elif WM_SP
            float* p_field = const_cast<float*>(inputs[i]->internalField().cdata());
#else
            double* p_field = const_cast<double*>(inputs[i]->internalField().cdata());
#endif
            std::copy_n(p_field, ndata, d->input_buffers[0].data() + i * ndata);
        }

        if (d->pd->set_node_data(d->inputs_names[0], d->input_buffers[0], d->layout, d->method)) {
            Foam::Info << "Succeeded set data" << Foam::endl;
        } else {
            Foam::Info << "Failed set data" << Foam::endl;
        }
    }

    Foam::Info << "Running Session...";
    d->pd->run();
    Foam::Info << "  Done! " << Foam::endl;

    if (outputs.size() == 1) {
#ifdef WM_DP
        double* p_field = const_cast<double*>(outputs[0]->internalField().cdata());
#elif WM_SP
        float* p_field = const_cast<float*>(outputs[0]->internalField().cdata());
#else
        double* p_field = const_cast<double*>(outputs[0]->internalField().cdata());
#endif
        
        if (d->pd->get_node_data(d->outputs_names[0], p_field, ndata, d->layout, d->method)) {
            Foam::Info << "Succeeded get data" << Foam::endl;
        } else {
            Foam::Info << "Failed get data" << Foam::endl;
        }
    } else {
        if (d->output_buffers.empty()) {
#ifdef WM_DP
            d->output_buffers.push_back(std::vector<double>(ndata * outputs.size()));
#elif WM_SP
            d->output_buffers.push_back(std::vector<float>(ndata * outputs.size()));
#else
            d->output_buffers.push_back(std::vector<double>(ndata * outputs.size()));
#endif
        } else if (d->output_buffers[0].size() != static_cast<unsigned int>(ndata * outputs.size())) {
            d->output_buffers[0].resize(ndata * outputs.size());
        }

        if (d->pd->get_node_data(d->outputs_names[0], d->output_buffers[0], d->layout, d->method)) {
            Foam::Info << "Succeeded get data" << Foam::endl;
        } else {
            Foam::Info << "Failed get data" << Foam::endl;
        }

        for (int i = 0; i < outputs.size(); i++) {
#ifdef WM_DP
            double* p_field = const_cast<double*>(outputs[i]->internalField().cdata());
#elif WM_SP
            float* p_field = const_cast<float*>(outputs[i]->internalField().cdata());
#else
            double* p_field = const_cast<double*>(outputs[i]->internalField().cdata());
#endif
            std::copy_n(d->output_buffers[0].data() + i * ndata, ndata, p_field);
        }
    }
}

void TF_OF_Predictor::predict(Foam::List<Foam::List<Foam::volScalarField*>>& multi_inputs,
                         Foam::List<Foam::List<Foam::volScalarField*>>& multi_outputs) {
    
    if ((multi_inputs.size() == 1) && (multi_outputs.size() == 1)) {
        this->predict(multi_inputs[0], multi_outputs[0]);
        return;
    }

    if (d->inputs_names.size()  == 0) {
        Foam::Info << "Input nodes not regestered " << Foam::endl;
        return;
    }
    if (d->outputs_names.size() == 0) {
        Foam::Info << "Output nodes not regestered " << Foam::endl;
        return;
    }

    for (int i = 0; i < multi_inputs.size(); i++) {
        for (int j = 0; j < multi_inputs[i].size(); j++) {
            if (multi_inputs[i][j]->size() == 0) {
                Foam::Info << "detected empty input data field, Failed to transfer input data" << Foam::endl;
                return;
            }
        }
    }

    for (int i = 0; i < multi_outputs.size(); i++) {
        for (int j = 0; j < multi_outputs[i].size(); j++) {
            if (multi_outputs[i][j]->size() == 0) {
                Foam::Info << "detected empty output data field, Failed to transfer output data" << Foam::endl;
                return;
            }
        }
    }


    if ((multi_inputs.size() != d->inputs_names.size()) && (multi_outputs.size() != d->outputs_names.size())) {
        Foam::Info << "Number of nodes checked failed: " << Foam::nl
                   << "Inputs:   Data: [" << multi_inputs.size() << "]  vs  Nodes: [" << d->inputs_names.size() << "]" << Foam::nl
                   << "Outputs:  Data: [" << multi_outputs.size() << "]  vs  Nodes: [" << d->outputs_names.size() << "]" << Foam::endl;
        return;
    }

    if (static_cast<unsigned int>(multi_inputs.size()) != d->input_buffers.size()) {
#ifdef WM_DP
        std::vector<std::vector<double>>(multi_inputs.size()).swap(d->input_buffers);
#elif WM_SP
        std::vector<std::vector<float>>(multi_inputs.size()).swap(d->input_buffers);
#else
        std::vector<std::vector<double>>(multi_inputs.size()).swap(d->input_buffers);
#endif
    }

    if (static_cast<unsigned int>(multi_outputs.size()) != d->output_buffers.size()) {
#ifdef WM_DP
        std::vector<std::vector<double>>(multi_outputs.size()).swap(d->output_buffers);
#elif WM_SP
        std::vector<std::vector<float>>(multi_outputs.size()).swap(d->output_buffers);
#else
        std::vector<std::vector<double>>(multi_outputs.size()).swap(d->output_buffers);
#endif
        
    }

    int ndata = multi_inputs[0][0]->size();
    if (d->pd->get_data_count() != ndata) {
        d->pd->set_data_count(ndata);
    }


    for (int i = 0; i < multi_inputs.size(); i++) {
        if (multi_inputs[i].size() == 1) {
#ifdef WM_DP
            if (d->pd->set_node_data(d->inputs_names[i], const_cast<double*>(multi_inputs[i][0]->internalField().cdata()), ndata, d->layout, d->method)) {
#elif
            if (d->pd->set_node_data(d->inputs_names[i], const_cast<float*>(multi_inputs[i][0]->internalField().cdata()), ndata, d->layout, d->method)) {
#else
            if (d->pd->set_node_data(d->inputs_names[i], const_cast<double*>(multi_inputs[i][0]->internalField().cdata()), ndata, d->layout, d->method)) {
#endif
                Foam::Info << "Succeeded set data" << Foam::endl;
            } else {
                Foam::Info << "Failed set data" << Foam::endl;
            }
        } else {
            if (d->input_buffers[i].size() != static_cast<unsigned int>(multi_inputs[i].size() * ndata)) {
                d->input_buffers[i].resize(multi_inputs[i].size() * ndata);
            }

            for (int j = 0; j < multi_inputs[i].size(); j++) {
#ifdef WM_DP
                std::copy_n(const_cast<double*>(multi_inputs[i][j]->internalField().cdata()), ndata, d->input_buffers[i].data() + j * ndata);
#elif WM_SP
                std::copy_n(const_cast<float*>(multi_inputs[i][j]->internalField().cdata()), ndata, d->input_buffers[i].data() + j * ndata);
#else
                std::copy_n(const_cast<double*>(multi_inputs[i][j]->internalField().cdata()), ndata, d->input_buffers[i].data() + j * ndata);
#endif
            }

            if (d->pd->set_node_data(d->inputs_names[i], d->input_buffers[i], d->layout, d->method)) {
                Foam::Info << "Succeeded set data" << Foam::endl;
            } else {
                Foam::Info << "Failed set data" << Foam::endl;
            }
        }
    }

    Foam::Info << "Running Session...";
    d->pd->run();
    Foam::Info << "  Done! " << Foam::endl;

    for (int i = 0; i < multi_outputs.size(); i++) {
        if (multi_outputs[i].size() == 1) {
#ifdef WM_DP
            if (d->pd->get_node_data(d->outputs_names[i], const_cast<double*>(multi_outputs[i][0]->internalField().cdata()), ndata, d->layout, d->method)) {
#elif
            if (d->pd->get_node_data(d->outputs_names[i], const_cast<float*>(multi_outputs[i][0]->internalField().cdata()), ndata, d->layout, d->method)) {
#else
            if (d->pd->get_node_data(d->outputs_names[i], const_cast<double*>(multi_outputs[i][0]->internalField().cdata()), ndata, d->layout, d->method)) {
#endif
                Foam::Info << "Succeeded set data" << Foam::endl;
            } else {
                Foam::Info << "Failed set data" << Foam::endl;
            }
        } else {
            if (d->output_buffers[i].size() != static_cast<unsigned int>(multi_outputs[i].size() * ndata)) {
                d->output_buffers[i].resize(multi_outputs[i].size() * ndata);
            }

            if (d->pd->get_node_data(d->outputs_names[i], d->output_buffers[i], d->layout, d->method)) {
                Foam::Info << "Succeeded set data" << Foam::endl;
            } else {
                Foam::Info << "Failed set data" << Foam::endl;
            }

            for (int j = 0; j < multi_outputs[i].size(); j++) {
#ifdef WM_DP
                std::copy_n(d->output_buffers[i].data() + j * ndata, ndata, const_cast<double*>(multi_outputs[i][j]->internalField().cdata()));
#elif WM_SP
                std::copy_n(d->output_buffers[i].data() + j * ndata, ndata, const_cast<float*>(multi_outputs[i][j]->internalField().cdata()));
#else
                std::copy_n(d->output_buffers[i].data() + j * ndata, ndata, const_cast<double*>(multi_outputs[i][j]->internalField().cdata()));
#endif
            }
        }
    }
}

void TF_OF_Predictor::predict(Foam::List<Foam::scalarField * > & inputs, Foam::List<Foam::scalarField * > & outputs) {

    if (d->inputs_names.size() == 0) {
        Foam::Info << "Input nodes not regestered " << Foam::endl;
        return;
    }
    if (d->outputs_names.size() == 0) {
        Foam::Info << "Output nodes not regestered " << Foam::endl;
        return;
    }
    if (d->inputs_names.size() != 1) {
        Foam::Info << "Not single input, please use predict(List<List<scalarField>>, List<List<scalarField>>) for multi in-out model." << Foam::endl;
        return;
    }
    if (d->outputs_names.size() != 1) {
        Foam::Info << "Not single output, please use predict(List<List<scalarField>>, List<List<scalarField>>) for multi in-out model." << Foam::endl;
        return;
    }

    if (inputs.size() == 0) {
        Foam::Info << "Empty input container" << Foam::endl;
        return;
    }

    if (outputs.size() == 0) {
        Foam::Info << "Empty output container" << Foam::endl;
        return;
    }

    int ndata = inputs[0]->size();
    if (d->pd->get_data_count() != ndata) {
        d->pd->set_data_count(ndata);
    }

    Foam::Info << "ndata is ::::" << ndata << Foam::endl;

    if (inputs.size() == 1) {
        if (d->pd->set_node_data(d->inputs_names[0], inputs[0]->data(), ndata, d->layout, d->method)) {
            Foam::Info << "Succeeded set data" << Foam::endl;
        } else {
            Foam::Info << "Failed set data" << Foam::endl;
        }
    } else {
        if (d->input_buffers.empty()) {
#ifdef WMDP
            d->input_buffers.push_back(std::vector<double>(ndata * inputs.size()));
#elif WM_SP
            d->input_buffers.push_back(std::vector<float>(ndata * inputs.size()));
#else
            d->input_buffers.push_back(std::vector<double>(ndata * inputs.size()));
#endif
            
        } else if (d->input_buffers[0].size() != static_cast<unsigned int>(ndata * inputs.size())) {
            d->input_buffers[0].resize(ndata * inputs.size());
        }

        for (int i = 0; i < inputs.size(); i++) {
            std::copy_n(inputs[i]->data(), ndata, d->input_buffers[0].data() + i * ndata);
        }

        if (d->pd->set_node_data(d->inputs_names[0], d->input_buffers[0], d->layout, d->method)) {
            Foam::Info << "Succeeded set data" << Foam::endl;
        } else {
            Foam::Info << "Failed set data" << Foam::endl;
        }
    }

    Foam::Info << "Running Session...";
    d->pd->run();
    Foam::Info << "  Done! " << Foam::endl;

    if (outputs.size() == 1) {
        if (d->pd->get_node_data(d->outputs_names[0], outputs[0]->data(), ndata, d->layout, d->method)) {
            Foam::Info << "Succeeded get data" << Foam::endl;
        } else {
            Foam::Info << "Failed get data" << Foam::endl;
        }
    } else {
        if (d->output_buffers.empty()) {
#ifdef WM_DP
            d->output_buffers.push_back(std::vector<double>(ndata * outputs.size()));
#elif WM_SP
            d->output_buffers.push_back(std::vector<float>(ndata * outputs.size()));
#else
            d->output_buffers.push_back(std::vector<double>(ndata * outputs.size()));
#endif
        } else if (d->output_buffers[0].size() != static_cast<unsigned int>(ndata * outputs.size())) {
            d->output_buffers[0].resize(ndata * outputs.size());
        }

        if (d->pd->get_node_data(d->outputs_names[0], d->output_buffers[0], d->layout, d->method)) {
            Foam::Info << "Succeeded get data" << Foam::endl;
        } else {
            Foam::Info << "Failed get data" << Foam::endl;
        }

        for (int i = 0; i < outputs.size(); i++) {
            std::copy_n(d->output_buffers[0].data() + i * ndata, ndata, outputs[i]->data());
        }
    }
}

void TF_OF_Predictor::predict(Foam::List<Foam::List<Foam::scalarField * > > & multi_inputs,
                              Foam::List<Foam::List<Foam::scalarField* > > & multi_outputs) {
    if ((multi_inputs.size() == 1) && (multi_outputs.size() == 1)) {
        this->predict(multi_inputs[0], multi_outputs[0]);
        return;
    }

    if (d->inputs_names.size() == 0) {
        Foam::Info << "Input nodes not regestered " << Foam::endl;
        return;
    }
    if (d->outputs_names.size() == 0) {
        Foam::Info << "Output nodes not regestered " << Foam::endl;
        return;
    }

    for (int i = 0; i < multi_inputs.size(); i++) {
        for (int j = 0; j < multi_inputs[i].size(); j++) {
            if (multi_inputs[i][j]->size() == 0) {
                Foam::Info << "detected empty input data field, Failed to transfer input data" << Foam::endl;
                return;
            }
        }
    }

    for (int i = 0; i < multi_outputs.size(); i++) {
        for (int j = 0; j < multi_outputs[i].size(); j++) {
            if (multi_outputs[i][j]->size() == 0) {
                Foam::Info << "detected empty output data field, Failed to transfer output data" << Foam::endl;
                return;
            }
        }
    }

    if ((multi_inputs.size() != d->inputs_names.size()) && (multi_outputs.size() != d->outputs_names.size())) {
        Foam::Info << "Number of nodes checked failed: " << Foam::nl
                   << "Inputs:   Data: [" << multi_inputs.size() << "]  vs  Nodes: [" << d->inputs_names.size() << "]" << Foam::nl
                   << "Outputs:  Data: [" << multi_outputs.size() << "]  vs  Nodes: [" << d->outputs_names.size() << "]" << Foam::endl;
        return;
    }

    if (static_cast<unsigned int>(multi_inputs.size()) != d->input_buffers.size()) {
#ifdef WM_DP
        std::vector<std::vector<double>>(multi_inputs.size()).swap(d->input_buffers);
#elif WM_SP
        std::vector<std::vector<float>>(multi_inputs.size()).swap(d->input_buffers);
#else
        std::vector<std::vector<double>>(multi_inputs.size()).swap(d->input_buffers);
#endif
        
    }

    if (static_cast<unsigned int>(multi_outputs.size()) != d->output_buffers.size()) {
#ifdef WM_DP
        std::vector<std::vector<double>>(multi_outputs.size()).swap(d->output_buffers);
#elif WM_SP
        std::vector<std::vector<float>>(multi_outputs.size()).swap(d->output_buffers);
#else
        std::vector<std::vector<double>>(multi_outputs.size()).swap(d->output_buffers);
#endif
        
    }

    int ndata = multi_inputs[0][0]->size();
    if (d->pd->get_data_count() != ndata) {
        d->pd->set_data_count(ndata);
    }

    for (int i = 0; i < multi_inputs.size(); i++) {
        if (multi_inputs[i].size() == 1) {
            if (d->pd->set_node_data(d->inputs_names[i], multi_inputs[i][0]->data(), ndata, d->layout, d->method)) {
                Foam::Info << "Succeeded set data" << Foam::endl;
            } else {
                Foam::Info << "Failed set data" << Foam::endl;
            }
        } else {
            if (d->input_buffers[i].size() != static_cast<unsigned int>(multi_inputs[i].size() * ndata)) {
                d->input_buffers[i].resize(multi_inputs[i].size() * ndata);
            }

            for (int j = 0; j < multi_inputs[i].size(); j++) {
                std::copy_n(multi_inputs[i][j]->data(), ndata, d->input_buffers[i].data() + j * ndata);
            }

            if (d->pd->set_node_data(d->inputs_names[i], d->input_buffers[i], d->layout, d->method)) {
                Foam::Info << "Succeeded set data" << Foam::endl;
            } else {
                Foam::Info << "Failed set data" << Foam::endl;
            }
        }
    }

    Foam::Info << "Running Session...";
    d->pd->run();
    Foam::Info << "  Done! " << Foam::endl;

    for (int i = 0; i < multi_outputs.size(); i++) {
        if (multi_outputs[i].size() == 1) {
            if (d->pd->get_node_data(d->outputs_names[i], multi_outputs[i][0]->data(), ndata, d->layout, d->method)) {
                Foam::Info << "Succeeded set data" << Foam::endl;
            } else {
                Foam::Info << "Failed set data" << Foam::endl;
            }
        } else {
            if (d->output_buffers[i].size() != static_cast<unsigned int>(multi_outputs[i].size() * ndata)) {
                d->output_buffers[i].resize(multi_outputs[i].size() * ndata);
            }

            if (d->pd->get_node_data(d->outputs_names[i], d->output_buffers[i], d->layout, d->method)) {
                Foam::Info << "Succeeded set data" << Foam::endl;
            } else {
                Foam::Info << "Failed set data" << Foam::endl;
            }

            for (int j = 0; j < multi_outputs[i].size(); j++) {
                std::copy_n(d->output_buffers[i].data() + j * ndata, ndata, multi_outputs[i][j]->data());
            }
        }
    }
}