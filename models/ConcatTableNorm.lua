local ConcatTableNorm, parent = torch.class('nn.ConcatTableNorm', 'nn.Container')

function ConcatTableNorm:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}
   self.n1 = torch.Tensor()
   self.n2 = torch.Tensor()
end

function ConcatTableNorm:updateOutput(input)
   for i=1,#self.modules do
      self.output[i] = self:rethrowErrors(self.modules[i], i, 'updateOutput', input)
   end
   return self.output
end

local function retable(t1, t2, f)
   for k, v in ipairs(t2) do
      if (torch.type(v) == "table") then
         t1[k] = retable(t1[k] or {}, t2[k], f)
      else
         f(t1, k, v)
      end
   end
   for i=#t2+1, #t1 do
      t1[i] = nil
   end
   return t1
end

local function backward(self, method, input, gradOutput, scale)
   local isTable = torch.type(input) == 'table'
   local wasTable = torch.type(self.gradInput) == 'table'
   if isTable then
      for i,module in ipairs(self.modules) do
         local currentGradInput = self:rethrowErrors(module, i, method, input, gradOutput[i], scale)
         if torch.type(currentGradInput) ~= 'table' then
            error"currentGradInput is not a table!"
         end
         if #input ~= #currentGradInput then
            error("table size mismatch: "..#input.." ~= "..#currentGradInput)
         end
         if i == 1 then
            self.gradInput = wasTable and self.gradInput or {}
            retable(self.gradInput, currentGradInput,
               function(t, k, v)
                  t[k] = t[k] or v:clone()
                  t[k]:resizeAs(v)
                  t[k]:copy(v)
               end
            )
         else
            retable(self.gradInput, currentGradInput,
               function(t, k, v)
                  if t[k] then
                     t[k]:add(v)
                  else
                     t[k] = v:clone()
                  end
               end
            )
         end
      end
   else
      if #self.modules ~= 2 then
        error("ConcatTableNorm can't be used with more or less than 2 modules.")
      end
      local gradInputs = {}
      self.gradInput = (not wasTable) and self.gradInput or input:clone()
      for i,module in ipairs(self.modules) do
         local currentGradInput = self:rethrowErrors(module, i, method, input, gradOutput[i], scale)
         gradInputs[i] = currentGradInput
         if i == 1 then
            self.gradInput:resizeAs(currentGradInput):copy(currentGradInput)
         else
            -- at this point, self.gradInput contains the gradInput of the conv
            -- currentGradInput point to the gradInput of the identity
            self.n1:resize(input:size(1), 1, 1, 1)
                   :copy(torch.pow(self.gradInput,2):sum(2):sum(3):sum(4):sqrt())
            self.n2:resize(input:size(1), 1, 1, 1)
                   :copy(torch.pow(currentGradInput,2):sum(2):sum(3):sum(4):sqrt())

            local n1_mean = self.n1:mean()
            local n2_mean = self.n2:mean()
            print(n1_mean, n2_mean)
            if n1_mean < n2_mean then
              self.gradInput:mul(n2_mean/n1_mean)
              self.n1:resize(input:size(1), 1, 1, 1)
                     :copy(torch.pow(self.gradInput,2):sum(2):sum(3):sum(4):sqrt())
              local new_n1_mean = self.n1:mean()
              print('->', new_n1_mean)
            end

            self.gradInput:add(currentGradInput)

         end
      end

   end
   return self.gradInput
end

function ConcatTableNorm:updateGradInput(input, gradOutput)
   return backward(self, 'updateGradInput', input, gradOutput)
end

function ConcatTableNorm:backward(input, gradOutput, scale)
   return backward(self, 'backward', input, gradOutput, scale)
end

function ConcatTableNorm:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i,module in ipairs(self.modules) do
      self:rethrowErrors(module, i, 'accGradParameters', input, gradOutput[i], scale)
   end
end

function ConcatTableNorm:accUpdateGradParameters(input, gradOutput, lr)
   for i,module in ipairs(self.modules) do
      self:rethrowErrors(module, i, 'accUpdateGradParameters', input, gradOutput[i], lr)
   end
end

function ConcatTableNorm:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local lastNext = '   `-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == #self.modules then
         str = str .. line .. tab .. lastNext .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end
