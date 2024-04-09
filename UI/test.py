from src.utils import load_model


def main():
    model_path = 'model_v1.pt'
    model = load_model(model_path)
    print(type(model.predict(["dick hui shit", 'You like CumGPT'])[0]))



if __name__ == '__main__':
    main()