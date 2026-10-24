import torch
import torch.nn.functional as F

class ActivationPatcher:
    def __init__(self, model):
        self.model = model
        self.activation_cache = {}
        self.hooks = []

    def _get_activation_hook(self, name):
        """Forward hook to cache intermediate layer/head activations."""
        def hook(module, input, output):
            # output shape typically (batch, seq_len, d_model) or attention weights
            self.activation_cache[name] = output.detach().clone()
        return hook

    def _patch_activation_hook(self, name, patched_tensor):
        """Forward hook to replace clean activations with patched ones."""
        def hook(module, input, output):
            return patched_tensor
        return hook

    def register_hooks(self, target_layer_name):
        """Registers a caching hook on a specific transformer sub-layer."""
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                h = module.register_forward_hook(self._get_activation_hook(name))
                self.hooks.append(h)

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activation_cache = {}

    def run_with_cache(self, x):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        return output

    def causal_patching_experiment(self, clean_input, corrupted_input, target_layer_name):
        """
        Performs causal activation patching:
        1. Run clean input to cache internal representations.
        2. Patch the clean representation into the forward pass of the corrupted input.
        3. Measure logit recovery.
        """
        # Step 1: Get clean activations
        self.register_hooks(target_layer_name)
        clean_logits = self.run_with_cache(clean_input)
        clean_activation = self.activation_cache[target_layer_name]
        self.clear_hooks()

        # Step 2: Setup hook to overwrite activations during corrupted run
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                h = module.register_forward_hook(self._patch_activation_hook(name, clean_activation))
                self.hooks.append(h)

        # Step 3: Run corrupted input with the patched activation
        self.model.eval()
        with torch.no_grad():
            patched_logits = self.model(corrupted_input)
        
        self.clear_hooks()
        
        # Calculate logit difference / recovery score
        recovery_score = F.softmax(patched_logits, dim=-1)
        return clean_logits, patched_logits, recovery_score