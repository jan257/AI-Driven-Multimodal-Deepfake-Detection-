from django.shortcuts import render, redirect # type: ignore
from .forms import UploadForm
from .deepfake_predictor import predict_deepfake
import os
from django.conf import settings # type: ignore
import time

def home(request):
    return render(request, 'detector/index.html')

def upload_and_predict(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['video']
            save_path = os.path.join(settings.MEDIA_ROOT, file.name)

            with open(save_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            # Store the path and redirect to loading screen
            request.session['media_path'] = file.name
            return redirect('loading')
    else:
        form = UploadForm()
    return render(request, 'detector/upload.html', {'form': form})

def loading(request):
    time.sleep(0)  # Simulate loading for 3 seconds
    return redirect('result')

def features(request):
    return render(request, 'detector/features.html')

def result(request):
    path = request.session.get('media_path')
    if not path:
        return redirect('upload')

    file_path = os.path.join(settings.MEDIA_ROOT, path)

    result_label, confidence, frame_confidences, gaze_confidences = predict_deepfake(file_path, is_video=True)

    return render(request, 'detector/result.html', {
    'result': result_label,
    'confidence': round(confidence * 100, 2),
    'media_url': f'/media/{path}',
    'frame_confidences': frame_confidences,
    'gaze_confidences': gaze_confidences,
})



