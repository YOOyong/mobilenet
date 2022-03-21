import os
from collections import Counter
import torch
from torchvision import datasets
from train_valid_loader import get_transforms

def init_mobilenet(num_classes = 3):
    # pretrained mobilenet 로드
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model= torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    
    # 전이학습을 위한 layer 고정
    for param in model.parameters():
        param.requires_grad = False

    # fc layer 재정의
    _num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(_num_ftrs, num_classes)
    
    return model.to(device)

def save_model(model, model_path, exp_summary):
    """
    state_dict 형식으로 저장
    """
    #폴더 생성
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    
    torch.save(model.state_dict(), os.path.join(model_path, f'mobilenet{exp_summary}.pt'))

def load_model(model_path, model_name):
    try:
        state_dict = torch.load(os.path.join(model_path, model_name))
    except:
        print("can not find model")
        exit()
    
    model = init_mobilenet()
    model.load_state_dict(state_dict)

    return model

def test_model(test_data_dir, model):
    """
    TEST_DATA_DIR
    -----------------------------
    ex) root
        |_ class1
        |   |_ img0.jpg
        |   |_ img1.jpg
        |   ...
        |_ class2
        |   |_ img0.jpg
        |   |_ img1.jpg
        |   ...
        ... 
    ------------------------------
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform, _ = get_transforms()
    try:
        test_dataset = datasets.ImageFolder(test_data_dir,transform = transform)
    except:
        print("can not find test data folder")
        exit()

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32) 

    class_count = dict(Counter(test_dataset.targets))
    class_names = test_dataset.classes

    total_corrects = 0
    class_corrects = {x : 0 for x in range(len(class_names))}

    #추론
    model.eval()

    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _ , pred = torch.max(outputs, 1)
        
        #클래스별 합
        for i in range(len(pred)):
            if pred[i] == labels.data[i]:
                class_corrects[pred[i].item()] += 1
        #전체 합
        total_corrects += torch.sum(pred == labels.data)

    #클래스별 점수
    for i in range(len(class_names)):
        print(f'{class_names[i]} acc : { class_corrects[i] / class_count[i] }')
    
    #전체 점수
    print(f'test data acc : {total_corrects / len(test_dataset)}')
        
