from django import forms # type: ignore

class UploadForm(forms.Form):
    video = forms.FileField(label="Upload Image or Video", widget=forms.ClearableFileInput(attrs={'accept': 'image/*,video/*'}))

