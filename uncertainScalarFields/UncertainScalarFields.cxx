#include "UncertainScalarFields.h"


//#include "stdfunc.h"
#include <iomanip>
#include <algorithm>

#include "vtkCellData.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyLine.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkProbeFilter.h"
#include "vtkDoubleArray.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkCallbackCommand.h"
#include "vtkImageWriter.h"
#include "vtkXMLImageDataWriter.h"
#include <sstream>
#include <random>
#include "math.h"

//blubb

vtkStandardNewMacro(UncertainScalarFields)

//-----------------------------------------------------------------------------
UncertainScalarFields::UncertainScalarFields()
{
    this->SetNumberOfOutputPorts(1);

    // by default process active point vectors
    this->SetInputArrayToProcess(0,0,0,vtkDataObject::FIELD_ASSOCIATION_POINTS, vtkDataSetAttributes::VECTORS);

    data = vtkSmartPointer<vtkImageData>::New();
}

//-----------------------------------------------------------------------------
UncertainScalarFields::~UncertainScalarFields()
{
}

//----------------------------------------------------------------------------
int UncertainScalarFields::FillInputPortInformation( int port, vtkInformation* info )
{ 
    info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);

    return 1;
}

//----------------------------------------------------------------------------
int UncertainScalarFields::FillOutputPortInformation( int port, vtkInformation* info )
{
    //info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkMultiBlockDataSet");
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
    return 1;
}

//----------------------------------------------------------------------------
int UncertainScalarFields::RequestUpdateExtent(vtkInformation*,vtkInformationVector** inputVector,vtkInformationVector* outputVector)
{
    vtkInformation *outInfo = outputVector->GetInformationObject(0);
    vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);

    int extent_grid[6];
    extent_grid[0] = 0;
    extent_grid[1] = resolution_grid[0] - 1;
    extent_grid[2] = 0;
    extent_grid[3] = resolution_grid[1] - 1;
    extent_grid[4] = 0;
    extent_grid[5] = resolution_grid[2] - 1;

    outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), extent_grid, 6);
    return 1;
}
int UncertainScalarFields::RequestInformation(vtkInformation *vtkNotUsed(request), vtkInformationVector **inputVector, vtkInformationVector *outputVector)
{
    vtkInformation *outInfo = outputVector->GetInformationObject(0);
    vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);

    int extent_grid[6];
    extent_grid[0] = 0;
    extent_grid[1] = resolution_grid[0] - 1;
    extent_grid[2] = 0;
    extent_grid[3] = resolution_grid[1] - 1;
    extent_grid[4] = 0;
    extent_grid[5] = resolution_grid[2] - 1;

    outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), extent_grid,6);
    outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), extent_grid,6);

    double spacing_grid[3];
    spacing_grid[0] = bounds_grid[0] / extent_grid[1];
    spacing_grid[1] = bounds_grid[1] / extent_grid[3];
    spacing_grid[2] = bounds_grid[2] / extent_grid[5];

    outInfo->Set(vtkDataObject::SPACING(), spacing_grid,3);
    outInfo->Set(vtkDataObject::ORIGIN(),origin_grid,3);
    return 1;
}

//----------------------------------------------------------------------------
int UncertainScalarFields::RequestData(vtkInformation *vtkNotUsed(request),
                         vtkInformationVector **inputVector,
                         vtkInformationVector *outputVector)
{
    bool is2D = true;
    int extent_grid[6];
    double spacing_grid[3];

    extent_grid[0] = origin_grid[0];
    extent_grid[1] = origin_grid[0] + (resolution_grid[0] - 1);
    extent_grid[2] = origin_grid[1];
    extent_grid[3] = origin_grid[1] + (resolution_grid[1] - 1);
    spacing_grid[0] = bounds_grid[0] / extent_grid[1];
    spacing_grid[1] = bounds_grid[1] / extent_grid[3];
    int scaleFactor = 1;


    if(bounds_grid[2] != 0){
        is2D = false;
        extent_grid[4] = origin_grid[2];
        extent_grid[5] = origin_grid[2] + (resolution_grid[2] - 1);
        spacing_grid[2] = bounds_grid[2] / extent_grid[5];
    } else {
        extent_grid[4] = extent_grid[5] = 0;
        spacing_grid[2] = 0.0;
        scaleFactor = resolution_grid[2];
        resolution_grid[2] = 1;
    }


    int array_len = resolution_grid[0]*resolution_grid[1]*resolution_grid[2];

    data->SetSpacing(spacing_grid);
    data->SetOrigin(origin_grid);
    data->SetExtent(extent_grid);

    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
    string file_name = "/export/home/ffallenb/Bachelorarbeit/ScalarFields/uncertain_scalar_field";
    uniform_real_distribution<double> random_value(noise[0], noise[1]);
    uniform_int_distribution<int> rand_int(0, 1);
    std::random_device rd;
    std::mt19937 re(rd());
    
    auto scalars = vtkSmartPointer<vtkDoubleArray>::New();
    scalars->SetNumberOfComponents(1);
    scalars->SetNumberOfTuples(array_len);
    scalars->SetName("data");

    if (is2D){
        for(int field = 0; field < numOfFields; field++){
            int tupleIndex = 0;

            /*double offsetX = random_value(re);
            double offsetY = pow((random_value(re) * 2), 2);*/
            double offsetX = 0.0;
            double offsetY = 0.0;

            if(shiftX) offsetX = random_value(re);
            if(shiftY) offsetY = random_value(re);

            for(int y = 0; y < resolution_grid[1]; y++){
                for(int x = 0; x < resolution_grid[0]; x++){
                    
                    double xPhys = double(x + offsetX) / double(resolution_grid[0] - 1) * 30.0;
                    double yPhys = double(y + offsetY) / double(resolution_grid[1] - 1) * 4 - 2;

                    double value = sin((2 * M_PI/ (resolution_grid[0] * scaleFactor)) * (((x - resolution_grid[0]/2) + offsetX) * (((y- (resolution_grid[1] / 2)) + offsetY)))); //oscillating field
                    
                    //double value = -fabs(yPhys - 0.5 * sin(xPhys*xPhys / 10.0)); //plotted sinus

                    //double value = xPhys - ((1/6) * pow(y, 2)) + 0.5 * cos(xPhys);

                    scalars ->SetTuple1(tupleIndex, value);
                    tupleIndex++;
                }
            }

            data->GetPointData()->SetScalars(scalars);

            stringstream index;
            index << "." << field << ".vtk";
            string path = file_name + string(index.str());
            writer->SetFileName(path.c_str());
            writer->SetInputData(data);
            writer->Write();
        }
    } else {
        for(int field = 0; field < numOfFields; field++){        
            
            int tupleIndex = 0;

            double offsetX = 0.0;
            double offsetY = 0.0;
            double offsetZ = 0.0;

            if(shiftX) offsetX = random_value(re);
            if(shiftY) offsetY = random_value(re);
            if(shiftZ) offsetZ = random_value(re);
            //int offsetZ = rand_int(re);
            
            //int breakZ = int(floor(resolution_grid[2] / 5));
            //int breakY = int(floor(resolution_grid[1] / 5));
            //int breakX = int(floor(resolution_grid[0] / 5));
            int sign = -1;
            double sigmaX = 1;
            double sigmaY = 1;
            double A = 1;
            double xNull = 0;
            double yNull = 0;
            
            for(int z = 0; z < resolution_grid[2]; z++){
                for(int y = 0; y < resolution_grid[1]; y++){
                    for(int x = 0; x < resolution_grid[0]; x++){

                        double xPhys = double(x + offsetX) / double(resolution_grid[0] - 1) * (4*M_PI); // - (0.5*M_PI);
                        double yPhys = double(y + offsetY) / double(resolution_grid[1] - 1) * (4*M_PI); // - (0.5*M_PI);
                        double zPhys = double(z + offsetZ) / double(resolution_grid[2] - 1) * (4*M_PI); // - (0.5*M_PI);

                        //double xPhys = double(x) / double(resolution_grid[0] - 1) * 6 - 3.;
                        //double yPhys = double(y) / double(resolution_grid[1] - 1) * 6 - 3.;
                        //double zPhys = double(z + offsetZ) / double(resolution_grid[2] - 1) * (8 * M_PI);

                        double value = cos(xPhys) + cos(yPhys) + cos(zPhys); //standard 3D cos set
                        //double value = (cos(((4*M_PI)/resolution_grid[0])*x) + cos(((4*M_PI)/resolution_grid[1])*y) + cos(((4*M_PI)/resolution_grid[2])*z)) + random_value(re);
                        //double value = (A * exp(-zPhys * ((pow((xPhys - xNull), 2)/2*pow(sigmaX, 2)) + (pow((yPhys - yNull), 2)/2*pow(sigmaY, 2))))); //2D gauss glocke in 3D

                        //double value = cos(zPhys);

                        scalars ->SetTuple1(tupleIndex, value);
                        //scalars ->SetTuple1(tupleIndex, double(tupleIndex));

                        tupleIndex++;
                    }
                }
            }
            data->GetPointData()->SetScalars(scalars);

            stringstream index;
            index << "." << field << ".vtk";
            string path = file_name + string(index.str());
            writer->SetFileName(path.c_str());
            writer->SetInputData(data);
            writer->Write();
        }
    }

    vtkInformation *outInfo = outputVector->GetInformationObject(0);
    
    vtkDataObject *output = vtkDataObject::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    output->DeepCopy(data);
    return 0;
}

void UncertainScalarFields::PrintSelf(ostream &os, vtkIndent indent)
{
}
