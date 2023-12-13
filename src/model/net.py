from torch import nn


class ContentEncoder(nn.Module):
    def __init__(self, in_feature: int = 512, out_feature: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, 2048)
        self.fc2 = nn.Linear(2048, out_feature)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
    

class StyleEncoder(nn.Module):
    def __init__(self, in_feature: int = 512, out_feature: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, out_feature)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += identity
        out = self.relu(out)
        out = self.fc3(out)
        
        return out
    

class GOYA(nn.Module):
    def __init__(self, content_encoder: ContentEncoder, style_encoder: StyleEncoder, n_features: int = 2048, projection_dim: int = 64):
        super().__init__()
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.n_features = n_features
        self.projector_c = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )
        self.projector_s = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )
        self.classifier_s = nn.Linear(self.n_features, 27)

        
    def forward(self, x):
        x_content = self.content_encoder(x)
        x_style = self.style_encoder(x)
        
        x_p_content = self.projector_c(x_content)
        x_p_style = self.projector_s(x_style)
        
        x_style_clf = self.classifier_s(x_style)

        return x_p_content, x_p_style, x_content, x_style, x_style_clf
    
    
def make_GOYA():
    content_encoder = ContentEncoder(512, 2048)
    style_encoder = StyleEncoder(512, 2048)
    model = GOYA(content_encoder, style_encoder, 2048, 64)
    
    return model


# ------- Other models ------- #

class Model(nn.Module):
    def __init__(self, content_encoder: ContentEncoder, style_encoder: StyleEncoder, n_features: int = 2048, projection_dim: int = 64):
        super().__init__()
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.n_features = n_features
        self.projector_c = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )
        self.projector_s = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )
        
    def forward(self, x):
        x_content = self.content_encoder(x)
        x_style = self.style_encoder(x)
        
        x_p_content = self.projector_c(x_content)
        x_p_style = self.projector_s(x_style)
        
        return x_p_content, x_p_style, x_content, x_style
    

def make_model_rm_clf():
    content_encoder = ContentEncoder(512, 2048)
    style_encoder = StyleEncoder(512, 2048)
    model = Model(content_encoder, style_encoder, 2048, 64)
    
    return model


class SingleLayer(nn.Module):
    def __init__(self, out_feature):
        super().__init__()
        self.fc = nn.Linear(512, out_feature)

    def forward(self, x):
        x = self.fc(x)
        
        return x
    
    
def make_model_single_layer(num_layer):
    content_encoder = SingleLayer(num_layer)
    style_encoder = SingleLayer(num_layer)
    model = Model(content_encoder, style_encoder, num_layer, 64)
    
    return model