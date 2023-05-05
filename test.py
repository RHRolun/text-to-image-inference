from inference import OVStableDiffusionPipeline, Vae_Decoder, Text_Encoder, Unet
from transformers import CLIPTokenizer
from diffusers.schedulers import LMSDiscreteScheduler

lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

vae_decoder = Vae_Decoder("https://vae-decoder-robert-serving-test.apps.rhods-internal.61tk.p1.openshiftapps.com/v2/models/vae-decoder/infer")
text_encoder = Text_Encoder("https://text-encoder2-robert-serving-test.apps.rhods-internal.61tk.p1.openshiftapps.com/v2/models/text-encoder2/infer")
unet_model = Unet("")
vae_encoder = None

ov_pipe = OVStableDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    unet=unet_model,
    vae_encoder=vae_encoder,
    vae_decoder=vae_decoder,
    scheduler=lms
)

text_prompt = "cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting, epic composition\nA golden daylight, hyper-realistic environment\nHyper and intricate detail, photo-realistic\nCinematic and volumetric light\nEpic concept art\nOctane render and Unreal Engine, trending on artstation"
num_steps = 20
seed = 42

result = ov_pipe(text_prompt, num_inference_steps=num_steps, seed=seed)

final_image = result['sample'][0]
final_image.save('result.png')